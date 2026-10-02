"""Prepare the weekly LAT all-sky archive for the moving-source pipeline.

The output is a resumable set of overlapping processing tiles. Candidate
counting remains non-overlapping: Stage A assigns each reconstructed track to
the nearest tile centre (a spherical Voronoi partition covering the full sky).
"""

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
import astropy.units as u
from scipy.spatial import SphericalVoronoi, cKDTree

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(PROJECT_ROOT))

from moving_utils_New import (  # noqa: E402
    ENERGY_MAX_MEV,
    ENERGY_MIN_MEV,
    SECONDS_PER_DAY,
    WINDOW_END,
    WINDOW_START,
    event_class_pass,
)

DEFAULT_ARCHIVE = PROJECT_ROOT / "data" / "fermi_allsky" / "weekly_photon"
DEFAULT_OUTPUT = PROJECT_ROOT / "allsky_queries" / "data"
WEEK_RE = re.compile(r"_w(?P<week>\d+)_p\d+_v(?P<version>\d+)\.fits$")
# The official HEASARC weekly-photon directory has no w512 product (it jumps
# from w511 to w513). This is a real no-product interval, not a local omission.
KNOWN_EMPTY_WEEK_NUMBERS = {512}
NULL_TIME_BIN_DAYS = 1.0
PREPARATION_SCHEMA_VERSION = 2


def canonical_hash(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def display_path(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def write_text_if_changed(path, text):
    path = Path(path)
    encoded = text.encode()
    if path.exists() and path.read_bytes() == encoded:
        return False
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)
    return True


def save_npz_if_changed(path, **arrays):
    path = Path(path)
    if path.exists():
        try:
            with np.load(path, allow_pickle=False) as existing:
                if set(existing.files) == set(arrays) and all(
                    np.array_equal(existing[name], value)
                    for name, value in arrays.items()
                ):
                    return False
        except (OSError, ValueError):
            pass
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)
    return True


def atomic_json(path, value):
    path = Path(path)
    return write_text_if_changed(path, json.dumps(value, indent=2) + "\n")


def gti_pass(times, starts, stops):
    starts = np.asarray(starts, dtype=float)
    stops = np.asarray(stops, dtype=float)
    order = np.argsort(starts)
    starts, stops = starts[order], stops[order]
    index = np.searchsorted(starts, times, side="right") - 1
    inside = index >= 0
    index = np.clip(index, 0, len(starts) - 1)
    inside &= times <= stops[index]
    return inside


def unit_vectors(ra_degrees, dec_degrees):
    ra = np.deg2rad(np.asarray(ra_degrees, dtype=float))
    dec = np.deg2rad(np.asarray(dec_degrees, dtype=float))
    cos_dec = np.cos(dec)
    return np.column_stack((
        cos_dec * np.cos(ra), cos_dec * np.sin(ra), np.sin(dec),
    ))


def detached_columns(hdu):
    """Copy FITS column definitions without retaining source data arrays."""
    attributes = (
        "name", "format", "unit", "null", "bscale", "bzero", "disp",
        "start", "dim", "coord_type", "coord_unit", "coord_ref_point",
        "coord_ref_value", "coord_inc", "time_ref_pos",
    )
    return fits.ColDefs([
        fits.Column(**{
            name: getattr(column, name)
            for name in attributes
            if getattr(column, name) is not None
        })
        for column in hdu.columns
    ])


def assign_table_rows(output_hdu, rows):
    """Populate a table, unpacking concatenated FITS X columns MSB first."""
    for name in output_hdu.columns.names:
        source = np.asarray(rows[name])
        target = output_hdu.data[name]
        if (
            target.ndim == 2
            and target.dtype == np.bool_
            and source.ndim == 2
            and np.issubdtype(source.dtype, np.integer)
            and source.dtype.itemsize == 1
        ):
            unpacked = np.unpackbits(
                source.astype(np.uint8, copy=False), axis=1, bitorder="big",
            )
            target[:] = unpacked[:, :target.shape[1]]
        else:
            target[:] = source


def fibonacci_centers(n_tiles):
    index = np.arange(n_tiles, dtype=float)
    z = 1.0 - 2.0 * (index + 0.5) / n_tiles
    longitude = np.mod(index * math.pi * (3.0 - math.sqrt(5.0)), 2.0 * math.pi)
    latitude = np.arcsin(z)
    galactic = SkyCoord(l=longitude * u.rad, b=latitude * u.rad, frame="galactic")
    icrs = galactic.icrs
    vectors = unit_vectors(icrs.ra.deg, icrs.dec.deg)
    voronoi = SphericalVoronoi(vectors)
    cell_radii = []
    for center, region in zip(vectors, voronoi.regions):
        dots = np.clip(voronoi.vertices[region] @ center, -1.0, 1.0)
        cell_radii.append(float(np.rad2deg(np.max(np.arccos(dots)))))
    return [
        {
            "tile_id": f"roi_{i + 1:03d}",
            "ra_degrees": float(icrs.ra.deg[i]),
            "dec_degrees": float(icrs.dec.deg[i]),
            "gal_l_degrees": float(galactic.l.deg[i]),
            "gal_b_degrees": float(galactic.b.deg[i]),
            "owner_cell_radius_degrees": cell_radii[i],
        }
        for i in range(n_tiles)
    ]


def scan_archive(archive):
    by_week = {}
    for path in sorted(Path(archive).glob("*.fits")):
        match = WEEK_RE.search(path.name)
        if not match:
            continue
        week, version = int(match["week"]), int(match["version"])
        current = by_week.get(week)
        if current is None or version > current[0]:
            with fits.open(path, memmap=True) as hdul:
                events = hdul["EVENTS"]
                header = events.header
                record = {
                    "path": str(path.resolve()),
                    "week": week,
                    "version": version,
                    "tstart": float(events.header["TSTART"]),
                    "tstop": float(events.header["TSTOP"]),
                    "nrows": int(header["NAXIS2"]),
                    "schema": [
                        (
                            header.get(f"TTYPE{index}"),
                            header.get(f"TFORM{index}"),
                            header.get(f"TDIM{index}"),
                        )
                        for index in range(1, int(header["TFIELDS"]) + 1)
                    ],
                    "size": path.stat().st_size,
                    "mtime_ns": path.stat().st_mtime_ns,
                    "fits_checksum_keywords": [
                        {
                            "hdu": hdu.name,
                            "checksum": hdu.header.get("CHECKSUM"),
                            "datasum": hdu.header.get("DATASUM"),
                        }
                        for hdu in hdul
                    ],
                }
            by_week[week] = (version, record)
    selected = sorted(
        (item[1] for item in by_week.values()
         if item[1]["tstop"] > WINDOW_START and item[1]["tstart"] < WINDOW_END),
        key=lambda item: item["tstart"],
    )
    if not selected:
        raise SystemExit(f"No weekly photon files overlap the analysis window in {archive}")
    schemas = {json.dumps(item["schema"]) for item in selected}
    if len(schemas) != 1:
        raise SystemExit("Weekly EVENTS schemas differ within the analysis window")
    if selected[0]["tstart"] > WINDOW_START or selected[-1]["tstop"] < WINDOW_END:
        raise SystemExit("Weekly archive does not bracket the full 2016-2026 window")
    expected_weeks = set(range(selected[0]["week"], selected[-1]["week"] + 1))
    missing_weeks = expected_weeks - {item["week"] for item in selected}
    unexpected = sorted(missing_weeks - KNOWN_EMPTY_WEEK_NUMBERS)
    if unexpected:
        raise SystemExit(f"Weekly archive is missing mission weeks: {unexpected}")
    print(
        f"Hashing {len(selected)} selected weekly FITS files for input integrity...",
        flush=True,
    )
    for index, item in enumerate(selected, start=1):
        item["sha256"] = file_sha256(item["path"])
        if index % 50 == 0 or index == len(selected):
            print(f"  hashed {index}/{len(selected)} source files", flush=True)
    return selected


def write_geometry(output_root, centers, processing_radius, config_hash):
    output_root.mkdir(parents=True, exist_ok=True)
    centers_path = output_root.parent / "tile_centers.csv"
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(centers[0]))
    writer.writeheader()
    writer.writerows(centers)
    write_text_if_changed(centers_path, buffer.getvalue())
    max_radius = max(item["owner_cell_radius_degrees"] for item in centers)
    null_model_path = output_root.parent / "null_time_model.npz"
    for center in centers:
        tile_dir = output_root / center["tile_id"]
        tile_dir.mkdir(parents=True, exist_ok=True)
        info = dict(center)
        info.update({
            "label": center["tile_id"],
            "query_radius_degrees": processing_radius,
            "search_radius_degrees": processing_radius,
            "analysis_radius_degrees": processing_radius,
            "processing_radius_degrees": processing_radius,
            "survey_geometry": "full_sky_nearest_center_voronoi",
            "sky_coverage_fraction": 1.0,
            "ownership_centers_file": display_path(centers_path),
            "ownership_tile_id": center["tile_id"],
            "ownership_tile_count": len(centers),
            "ownership_max_radius_degrees": max_radius,
            "null_time_model_file": display_path(null_model_path),
            "preparation_config_sha256": config_hash,
        })
        atomic_json(tile_dir / "query_info.json", info)
    return centers_path, max_radius


def block_is_complete(manifest_path, fingerprint, deep=False):
    try:
        manifest = json.loads(Path(manifest_path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    if manifest.get("fingerprint") != fingerprint:
        return False
    outputs_valid = all(
        Path(item["path"]).exists()
        and Path(item["path"]).stat().st_size == item["size"]
        and (not deep or file_sha256(item["path"]) == item.get("sha256"))
        for item in manifest.get("outputs", [])
    )
    histogram = manifest.get("null_histogram", {})
    histogram_path = Path(histogram.get("path", ""))
    histogram_valid = (
        histogram_path.is_file()
        and histogram_path.stat().st_size == histogram.get("size")
        and (not deep or file_sha256(histogram_path) == histogram.get("sha256"))
    )
    return outputs_valid and histogram_valid


def process_block(task):
    block_index = task["block_index"]
    block_start, block_stop = task["block_start"], task["block_stop"]
    output_root = Path(task["output_root"])
    centers = task["centers"]
    center_vectors = unit_vectors(
        [item["ra_degrees"] for item in centers],
        [item["dec_degrees"] for item in centers],
    )
    center_tree = cKDTree(center_vectors)
    null_edges = np.arange(
        float(WINDOW_START), float(WINDOW_END),
        NULL_TIME_BIN_DAYS * SECONDS_PER_DAY,
    )
    if null_edges[-1] < WINDOW_END:
        null_edges = np.append(null_edges, float(WINDOW_END))
    else:
        null_edges[-1] = float(WINDOW_END)
    null_counts = np.zeros((len(centers), len(null_edges) - 1), dtype=np.int64)
    chord_radius = 2.0 * math.sin(math.radians(task["processing_radius"]) / 2.0)
    source_files = [
        item for item in task["archive_files"]
        if item["tstop"] > block_start and item["tstart"] < block_stop
    ]
    fingerprint = canonical_hash({
        "config": task["config_hash"],
        "block": [block_start, block_stop],
        "sources": [
            (item["path"], item["sha256"])
            for item in source_files
        ],
    })
    manifest_path = output_root / "_blocks" / f"block_{block_index:03d}.json"
    if block_is_complete(manifest_path, fingerprint):
        return json.loads(manifest_path.read_text())

    rows_by_tile = [[] for _ in centers]
    gti_parts = []
    template = None
    rows_read = rows_selected = 0
    source_integrity = []
    for source in source_files:
        with fits.open(source["path"], memmap=True, checksum=True) as hdul:
            events_hdu = hdul["EVENTS"]
            events = events_hdu.data
            gti = hdul["GTI"].data
            if template is None:
                template = (
                    hdul[0].header.copy(), events_hdu.header.copy(),
                    hdul["GTI"].header.copy(), events.dtype, gti.dtype,
                    detached_columns(events_hdu),
                    detached_columns(hdul["GTI"]),
                )
            times = np.asarray(events["TIME"], dtype=float)
            mask = (
                (times >= block_start) & (times < block_stop)
                & (times > WINDOW_START) & (times < WINDOW_END)
                & (np.asarray(events["ENERGY"], dtype=float) >= ENERGY_MIN_MEV)
                & (np.asarray(events["ENERGY"], dtype=float) < ENERGY_MAX_MEV)
                & (np.asarray(events["ZENITH_ANGLE"], dtype=float) < task["zmax"])
                & event_class_pass(events["EVENT_CLASS"], task["event_class_bit"])
                & gti_pass(times, gti["START"], gti["STOP"])
                & np.isfinite(times)
                & np.isfinite(np.asarray(events["ENERGY"], dtype=float))
                & np.isfinite(np.asarray(events["RA"], dtype=float))
                & np.isfinite(np.asarray(events["DEC"], dtype=float))
            )
            rows_read += len(events)
            selected = events[np.flatnonzero(mask)]
            rows_selected += len(selected)
            if len(selected):
                selected_vectors = unit_vectors(selected["RA"], selected["DEC"])
                tree = cKDTree(selected_vectors)
                memberships = tree.query_ball_point(center_vectors, chord_radius)
                for tile_index, indices in enumerate(memberships):
                    if indices:
                        rows_by_tile[tile_index].append(selected[np.asarray(indices)])
                _, owner_indices = center_tree.query(selected_vectors, k=1)
                time_indices = np.searchsorted(
                    null_edges, np.asarray(selected["TIME"], dtype=float), side="right",
                ) - 1
                time_indices = np.clip(time_indices, 0, len(null_edges) - 2)
                flat = owner_indices * (len(null_edges) - 1) + time_indices
                null_counts += np.bincount(
                    flat, minlength=null_counts.size,
                ).reshape(null_counts.shape)
            gti_mask = (gti["STOP"] >= block_start) & (gti["START"] < block_stop)
            if np.any(gti_mask):
                clipped = gti[np.flatnonzero(gti_mask)].copy()
                clipped["START"] = np.maximum(clipped["START"], block_start)
                clipped["STOP"] = np.minimum(clipped["STOP"], block_stop)
                gti_parts.append(clipped)
            checks = []
            for hdu in hdul:
                checksum_ok = int(hdu.verify_checksum())
                datasum_ok = int(hdu.verify_datasum())
                if checksum_ok != 1 or datasum_ok != 1:
                    raise RuntimeError(
                        f"FITS checksum failure in {source['path']} HDU {hdu.name}"
                    )
                checks.append({
                    "hdu": hdu.name,
                    "checksum_valid": True,
                    "datasum_valid": True,
                })
            source_integrity.append({"path": source["path"], "hdus": checks})

    if template is None:
        raise RuntimeError(f"Block {block_index} has no source weekly files")
    (
        primary_header, events_header, gti_header, events_dtype, gti_dtype,
        events_columns, gti_columns,
    ) = template
    combined_gti = (
        np.concatenate(gti_parts) if gti_parts else np.empty(0, dtype=gti_dtype)
    )
    outputs = []
    for tile_index, center in enumerate(centers):
        data = (
            np.concatenate(rows_by_tile[tile_index])
            if rows_by_tile[tile_index] else np.empty(0, dtype=events_dtype)
        )
        destination = (
            output_root / center["tile_id"] / f"events_block_{block_index:03d}.fits"
        )
        temporary = destination.with_suffix(".fits.tmp")
        event_header = events_header.copy()
        gti_out_header = gti_header.copy()
        for header in (event_header, gti_out_header):
            header["TSTART"] = block_start
            header["TSTOP"] = block_stop
        events_out = fits.BinTableHDU.from_columns(
            events_columns, nrows=len(data), header=event_header, name="EVENTS",
        )
        assign_table_rows(events_out, data)
        gti_out = fits.BinTableHDU.from_columns(
            gti_columns, nrows=len(combined_gti), header=gti_out_header, name="GTI",
        )
        assign_table_rows(gti_out, combined_gti)
        hdul_out = fits.HDUList([
            fits.PrimaryHDU(header=primary_header.copy()),
            events_out,
            gti_out,
        ])
        hdul_out.writeto(temporary, overwrite=True, checksum=True)
        temporary.replace(destination)
        outputs.append({
            "tile_id": center["tile_id"],
            "path": str(destination.resolve()),
            "size": destination.stat().st_size,
            "nrows": int(len(data)),
            "sha256": file_sha256(destination),
        })
    histogram_path = output_root / "_blocks" / f"null_histogram_{block_index:03d}.npz"
    save_npz_if_changed(
        histogram_path,
        time_edges_met=null_edges,
        counts=null_counts,
    )
    histogram_record = {
        "path": str(histogram_path.resolve()),
        "size": histogram_path.stat().st_size,
        "sha256": file_sha256(histogram_path),
    }
    manifest = {
        "block_index": block_index,
        "block_start_met": block_start,
        "block_stop_met": block_stop,
        "fingerprint": fingerprint,
        "source_weeks": [item["week"] for item in source_files],
        "raw_rows_read": rows_read,
        "quality_selected_rows_before_spatial_duplication": rows_selected,
        "tile_rows_written": sum(item["nrows"] for item in outputs),
        "source_fits_integrity": source_integrity,
        "null_histogram": histogram_record,
        "outputs": outputs,
    }
    atomic_json(manifest_path, manifest)
    return manifest


def block_tasks(args, centers, archive_files, config_hash):
    block_seconds = args.block_days * SECONDS_PER_DAY
    edges = [float(WINDOW_START)]
    while edges[-1] < WINDOW_END:
        edges.append(min(float(WINDOW_END), edges[-1] + block_seconds))
    return [
        {
            "block_index": index,
            "block_start": edges[index],
            "block_stop": edges[index + 1],
            "output_root": str(project_path(args.output_root).resolve()),
            "centers": centers,
            "archive_files": archive_files,
            "processing_radius": args.processing_radius,
            "zmax": args.zmax,
            "event_class_bit": args.event_class_bit,
            "config_hash": config_hash,
        }
        for index in range(len(edges) - 1)
    ]


def prepare(args):
    archive = project_path(args.archive)
    output_root = project_path(args.output_root)
    if args.workers < 1 or args.block_days <= 0 or args.processing_radius <= 0:
        raise SystemExit("workers, block-days, and processing-radius must be positive")
    print(f"Scanning weekly archive: {archive}", flush=True)
    archive_files = scan_archive(archive)
    centers = fibonacci_centers(args.n_tiles)
    max_owner_radius = max(item["owner_cell_radius_degrees"] for item in centers)
    if max_owner_radius > args.max_owner_radius and not args.allow_incomplete_coverage:
        raise SystemExit(
            f"{args.n_tiles} tiles have max owner radius {max_owner_radius:.3f} deg; "
            f"require <= {args.max_owner_radius:.3f} deg"
        )
    config = {
        "schema_version": PREPARATION_SCHEMA_VERSION,
        "prepared_fits_schema": "preserve_source_column_tforms_v2",
        "events_schema": archive_files[0]["schema"],
        "window_met": [WINDOW_START, WINDOW_END],
        "energy_mev": [ENERGY_MIN_MEV, ENERGY_MAX_MEV],
        "zmax_degrees": args.zmax,
        "event_class_bit": args.event_class_bit,
        "n_tiles": args.n_tiles,
        "processing_radius_degrees": args.processing_radius,
        "max_owner_radius_degrees": max_owner_radius,
        "block_days": args.block_days,
        "archive": [
            {key: item[key] for key in (
                "path", "week", "version", "tstart", "tstop", "size",
                "sha256", "fits_checksum_keywords",
            )}
            for item in archive_files
        ],
        "known_no_product_weeks": sorted(KNOWN_EMPTY_WEEK_NUMBERS),
    }
    config_hash = canonical_hash(config)
    (output_root / "_blocks").mkdir(parents=True, exist_ok=True)
    centers_path, max_owner_radius = write_geometry(
        output_root, centers, args.processing_radius, config_hash,
    )
    tasks = block_tasks(args, centers, archive_files, config_hash)
    print(
        f"Preparing {len(archive_files)} weekly files into {len(centers)} tiles "
        f"and {len(tasks)} resumable blocks (owner radius <= {max_owner_radius:.3f} deg).",
        flush=True,
    )
    free_tb = shutil.disk_usage(output_root).free / 10**12
    print(f"Free space at destination: {free_tb:.2f} TB", flush=True)
    if args.workers == 1:
        manifests = []
        for task in tasks:
            result = process_block(task)
            manifests.append(result)
            print(
                f"block {result['block_index'] + 1}/{len(tasks)} ready: "
                f"{result['tile_rows_written']:,} tile rows",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            manifests = list(executor.map(process_block, tasks))
    manifests.sort(key=lambda item: item["block_index"])
    null_counts = None
    null_edges = None
    for item in manifests:
        with np.load(item["null_histogram"]["path"], allow_pickle=False) as block_null:
            block_edges = block_null["time_edges_met"]
            block_counts = block_null["counts"]
        if null_edges is None:
            null_edges = block_edges
            null_counts = np.zeros_like(block_counts, dtype=np.int64)
        if not np.array_equal(null_edges, block_edges):
            raise RuntimeError("Null-time histogram edges differ between blocks")
        null_counts += block_counts
    null_model_path = output_root.parent / "null_time_model.npz"
    save_npz_if_changed(
        null_model_path,
        time_edges_met=null_edges,
        counts=null_counts,
        center_vectors=unit_vectors(
            [item["ra_degrees"] for item in centers],
            [item["dec_degrees"] for item in centers],
        ),
        tile_ids=np.asarray([item["tile_id"] for item in centers]),
    )
    null_model_sha256 = file_sha256(null_model_path)
    for center in centers:
        files = [
            output_root / center["tile_id"] / f"events_block_{item['block_index']:03d}.fits"
            for item in manifests
        ]
        write_text_if_changed(
            output_root / center["tile_id"] / "events.txt",
            "\n".join(display_path(path) for path in files) + "\n",
        )
    identity_payload = {
        "preparation_config_sha256": config_hash,
        "centers_sha256": hashlib.sha256(centers_path.read_bytes()).hexdigest(),
        "null_time_model_sha256": null_model_sha256,
        "blocks": [
            {
                "fingerprint": item["fingerprint"],
                "null_histogram_sha256": item["null_histogram"]["sha256"],
                "output_sha256": [output["sha256"] for output in item["outputs"]],
            }
            for item in manifests
        ],
    }
    dataset_sha256 = canonical_hash(identity_payload)
    existing_manifest_path = output_root / "_dataset_manifest.json"
    existing_created = None
    if existing_manifest_path.exists():
        try:
            existing = json.loads(existing_manifest_path.read_text())
            if existing.get("dataset_sha256") == dataset_sha256:
                existing_created = existing.get("created_utc")
        except json.JSONDecodeError:
            pass
    dataset_manifest = {
        "status": "complete",
        "created_utc": existing_created or datetime.now(timezone.utc).isoformat(),
        "dataset_sha256": dataset_sha256,
        "identity_payload": identity_payload,
        "preparation_config_sha256": config_hash,
        "config": config,
        "centers_file": display_path(centers_path),
        "centers_sha256": hashlib.sha256(centers_path.read_bytes()).hexdigest(),
        "null_time_model_file": display_path(null_model_path),
        "null_time_model_sha256": null_model_sha256,
        "null_time_model": (
            "owner-cell empirical daily time bootstrap keyed by RUN_ID/EVENT_ID"
        ),
        "survey_geometry": "full_sky_nearest_center_voronoi",
        "sky_coverage_fraction": 1.0,
        "n_blocks": len(manifests),
        "raw_rows_read_across_blocks": sum(item["raw_rows_read"] for item in manifests),
        "quality_selected_rows": sum(
            item["quality_selected_rows_before_spatial_duplication"] for item in manifests
        ),
        "tile_rows_written": sum(item["tile_rows_written"] for item in manifests),
        "block_manifests": [
            display_path(output_root / "_blocks" / f"block_{item['block_index']:03d}.json")
            for item in manifests
        ],
    }
    atomic_json(output_root / "_dataset_manifest.json", dataset_manifest)
    verify_dataset(
        output_root,
        source_hashes={item["path"]: item["sha256"] for item in archive_files},
    )
    print(f"Full-sky prepared dataset is ready under {output_root}", flush=True)


def verify_dataset(output_root, deep=True, source_hashes=None):
    output_root = project_path(output_root)
    manifest_path = output_root / "_dataset_manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"Prepared dataset is incomplete: no {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "complete":
        raise SystemExit("Prepared dataset manifest is not marked complete")
    schema_version = manifest.get("config", {}).get("schema_version")
    if schema_version != PREPARATION_SCHEMA_VERSION:
        raise SystemExit(
            f"Prepared dataset schema v{schema_version} is obsolete; rerun "
            "the prepare command to rebuild schema-preserving FITS tiles"
        )
    n_tiles = int(manifest["config"]["n_tiles"])
    n_blocks = int(manifest["n_blocks"])
    centers_path = project_path(manifest["centers_file"])
    centers_hash = hashlib.sha256(centers_path.read_bytes()).hexdigest()
    if centers_hash != manifest.get("centers_sha256"):
        raise SystemExit("Tile-centre ownership file changed after preparation")
    if canonical_hash(manifest.get("identity_payload")) != manifest.get("dataset_sha256"):
        raise SystemExit("Prepared dataset scientific identity is invalid")
    if deep:
        source_hashes = source_hashes or {}
        for source in manifest["config"]["archive"]:
            source_path = Path(source["path"])
            if not source_path.is_file() or source_path.stat().st_size != source["size"]:
                raise SystemExit(f"Missing or size-changed source FITS file: {source_path}")
            actual_hash = source_hashes.get(str(source_path))
            if actual_hash is None:
                actual_hash = file_sha256(source_path)
            if actual_hash != source.get("sha256"):
                raise SystemExit(f"Source FITS SHA-256 mismatch: {source_path}")
    null_model_path = project_path(manifest["null_time_model_file"])
    if file_sha256(null_model_path) != manifest.get("null_time_model_sha256"):
        raise SystemExit("Null-time model changed after preparation")
    tile_dirs = sorted(path for path in output_root.glob("roi_*") if path.is_dir())
    if len(tile_dirs) != n_tiles:
        raise SystemExit(f"Expected {n_tiles} tile directories; found {len(tile_dirs)}")
    for block_path_string in manifest["block_manifests"]:
        block_path = project_path(block_path_string)
        block = json.loads(block_path.read_text())
        if len(block.get("outputs", [])) != n_tiles:
            raise SystemExit(f"Prepared block has the wrong tile count: {block_path}")
        if not block_is_complete(block_path, block["fingerprint"], deep=deep):
            raise SystemExit(f"Incomplete or changed prepared block: {block_path}")
        sample_output = Path(block["outputs"][0]["path"])
        with fits.open(sample_output, memmap=True) as hdul:
            header = hdul["EVENTS"].header
            actual_schema = [
                [
                    header.get(f"TTYPE{index}"),
                    header.get(f"TFORM{index}"),
                    header.get(f"TDIM{index}"),
                ]
                for index in range(1, int(header["TFIELDS"]) + 1)
            ]
        if actual_schema != manifest["config"]["events_schema"]:
            raise SystemExit(
                f"Prepared EVENTS schema differs from source schema: {sample_output}"
            )
    for tile_dir in tile_dirs:
        info = json.loads((tile_dir / "query_info.json").read_text())
        files = [line for line in (tile_dir / "events.txt").read_text().splitlines() if line]
        if info.get("ownership_tile_id") != tile_dir.name or len(files) != n_blocks:
            raise SystemExit(f"Invalid tile metadata or manifest: {tile_dir}")
        for value in files:
            if not project_path(value).exists():
                raise SystemExit(f"Missing prepared FITS file: {value}")
    with np.load(null_model_path, allow_pickle=False) as null_model:
        counts = null_model["counts"]
        edges = null_model["time_edges_met"]
        model_tiles = null_model["tile_ids"]
    if counts.shape != (n_tiles, len(edges) - 1) or len(model_tiles) != n_tiles:
        raise SystemExit("Null-time model dimensions do not match the survey geometry")
    if int(counts.sum()) != int(manifest["quality_selected_rows"]):
        raise SystemExit("Null-time model event count disagrees with preparation")
    print(
        f"Verified {n_tiles} full-sky tiles, {n_blocks} blocks/tile, "
        f"coverage={manifest['sky_coverage_fraction']:.1f}, "
        f"integrity={'SHA-256' if deep else 'size-only'}.",
        flush=True,
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--archive", default=str(DEFAULT_ARCHIVE))
    prepare_parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    prepare_parser.add_argument("--n-tiles", type=int, default=448)
    prepare_parser.add_argument("--processing-radius", type=float, default=18.0)
    prepare_parser.add_argument("--max-owner-radius", type=float, default=7.5)
    prepare_parser.add_argument("--block-days", type=float, default=91.25)
    prepare_parser.add_argument("--workers", type=int, default=1)
    prepare_parser.add_argument("--zmax", type=float, default=90.0)
    prepare_parser.add_argument("--event-class-bit", type=int, default=128)
    prepare_parser.add_argument("--allow-incomplete-coverage", action="store_true")
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    verify_parser.add_argument("--fast", action="store_true",
                               help="check sizes/manifests without rehashing FITS files")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "prepare":
        prepare(args)
    else:
        verify_dataset(args.output_root, deep=not args.fast)


if __name__ == "__main__":
    main()
