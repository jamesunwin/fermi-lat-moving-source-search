import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_ROOT = Path(
    os.environ.get("MOVING_RESULTS_ROOT", PROJECT_ROOT / "moving" / "results_v5")
)
if not DEFAULT_RESULTS_ROOT.is_absolute():
    DEFAULT_RESULTS_ROOT = PROJECT_ROOT / DEFAULT_RESULTS_ROOT
DEFAULT_ROI_ROOT = PROJECT_ROOT / "allsky_queries" / "data"
DEFAULT_OUTPUT_ROOT = DEFAULT_RESULTS_ROOT / "roi_queries_all"
DEFAULT_NONEMPTY_ROOT = (
    DEFAULT_RESULTS_ROOT / "roi_queries_nonempty"
)

# drive the updated search
# script. ---
FPS_SCRIPT = PROJECT_ROOT / "moving" / "fps_moving_v5_New.py"
STAGE_A_CODE_FILES = (
    FPS_SCRIPT,
    PROJECT_ROOT / "moving_utils_New.py",
    PROJECT_ROOT / "moving" / "coherent_null.py",
    PROJECT_ROOT / "moving" / "analysis_region_v5.py",
    PROJECT_ROOT / "moving" / "adaptive_seeding_v5.py",
)
_CONTENT_SIGNATURE_CACHE = {}



def parse_args():
    parser = argparse.ArgumentParser(
        description = "Run fps_moving_v5_New.py across downloaded ROI data."
    )
    parser.add_argument("--roi-root", type = str, default = str(DEFAULT_ROI_ROOT))
    parser.add_argument(
        "--roi-ids", default=None,
        help="optional comma-separated subset (diagnostic campaigns only)",
    )
    parser.add_argument(
        "--output-root", type = str, default = str(DEFAULT_OUTPUT_ROOT)
    )
    parser.add_argument(
        "--nonempty-root", type = str, default = str(DEFAULT_NONEMPTY_ROOT)
    )
    parser.add_argument("--bin-time-days", type = float, default = None)
    parser.add_argument("--min-track-length", type = int, default = None)
    # 0.60 rejected perfect constant-
    # velocity tracks (centroid noise dominates step ratios); see fps script.
    parser.add_argument("--min-velocity-consistency", type = float, default = None)
    parser.add_argument("--save-pre-tube", action = "store_true")
    parser.add_argument("--save-bin-candidates", action="store_true")
    # final Stage-A stationary control. ---
    parser.add_argument(
        "--stationary-validation",
        action="store_true",
        help=(
            "save final per-bin Stage-A detections, disable the catalogue "
            "on-source veto, and bypass motion linking"
        ),
    )

    parser.add_argument("--skip-existing", action = "store_true")
    parser.add_argument(
        "--workers", type=int,
        default=int(os.environ.get("FPS_BATCH_WORKERS", "1")),
        help="number of independent ROI searches to run concurrently",
    )

    # expose the
    # new fps_moving_v5_New controls per batch run: link rate in deg/yr,
    # per-track skip budget, newer catalog FITS for masking, time-scramble
    # null-calibration mode, and the motion-filter toggle for the slow-mover
    # / parallax mode with sub-annual bins. ---
    parser.add_argument("--max-link-deg-per-year", type = float, default = None)
    parser.add_argument("--max-total-skipped-bins", type = int, default = None)
    parser.add_argument("--max-link-states-per-skip", type = int, default = None)
    parser.add_argument(
        "--catalog-fits", type = str, default = None,
        help = "Path to a FL16Y catalog FITS file for masking.",
    )
    parser.add_argument(
        "--scramble-times", action = "store_true",
        help = (
            "Permute photon arrival times in every ROI to measure the "
            "false-track rate (threshold calibration)."
        ),
    )
    parser.add_argument("--scramble-seed", type = int, default = 0)
    parser.add_argument(
        "--disable-motion-filter", action = "store_true",
        help = "Skip the straight-tube filters (slow-mover/parallax mode).",
    )


    # expose the seeding operating point. If not
    # given, any FPS_* values already in the environment pass through, so
    # run_calibration_v5.py controls the operating point via env. ---
    parser.add_argument("--min-samples", type = int, default = None,
                        help = "DBSCAN min_samples floor (default 4).")
    parser.add_argument("--eps-scale", type = float, default = None,
                        help = "Scale factor on the PSF-based eps (default 1).")
    parser.add_argument("--min-sigma", type = float, default = None,
                        help = "Li-Ma candidate threshold (default 2.5).")

    # shared BUFFER-region Monte Carlo areas and
    # catalogue-annulus masking controls. ---
    parser.add_argument("--mc-area-samples", type=int, default=None)
    parser.add_argument("--max-mc-area-samples", type=int, default=None)
    parser.add_argument(
        "--max-mc-to-poisson-ratio", type=float, default=None,
    )
    parser.add_argument(
        "--disable-annulus-mask", action="store_true",
        help="geometry-only attribution run; use the fixed on mask",
    )
    parser.add_argument("--catalog-mask-r68-scale", type=float, default=None)
    parser.add_argument("--catalog-mask-min-radius", type=float, default=None)
    parser.add_argument("--catalog-mask-min-flux", type=float, default=None)
    parser.add_argument(
        "--min-unmasked-annulus-fraction", type=float, default=None,
    )
    parser.add_argument("--min-unmasked-off-counts", type=int, default=None)

    return parser.parse_args()


def project_path(path_string):
    path = Path(path_string)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_roi_info(roi_dir):
    return json.loads((roi_dir / "query_info.json").read_text())


def relative_output_label(output_dir):
    results_root = DEFAULT_RESULTS_ROOT
    try:
        return str(output_dir.relative_to(results_root))
    except ValueError:
        return str(output_dir.resolve())


def build_run_environment(args, roi_dir, roi_info, output_dir):
    run_env = os.environ.copy()
    optional_settings = {
        "FPS_BIN_TIME_DAYS": args.bin_time_days,
        "FPS_MIN_TRACK_LENGTH": args.min_track_length,
        "FPS_MIN_VELOCITY_CONSISTENCY": args.min_velocity_consistency,
        "FPS_MAX_LINK_DEG_PER_YEAR": args.max_link_deg_per_year,
        "FPS_MAX_TOTAL_SKIPPED_BINS": args.max_total_skipped_bins,
        "FPS_MAX_LINK_STATES_PER_SKIP": args.max_link_states_per_skip,
        "FPS_MIN_SAMPLES": args.min_samples,
        "FPS_EPS_SCALE": args.eps_scale,
        "FPS_MIN_SIGMA": args.min_sigma,
        "FPS_MC_AREA_SAMPLES": args.mc_area_samples,
        "FPS_MAX_MC_AREA_SAMPLES": args.max_mc_area_samples,
        "FPS_MAX_MC_TO_POISSON_RATIO": args.max_mc_to_poisson_ratio,
        "FPS_CATALOG_MASK_R68_SCALE": args.catalog_mask_r68_scale,
        "FPS_CATALOG_MASK_MIN_RADIUS_DEG": args.catalog_mask_min_radius,
        "FPS_CATALOG_MASK_MIN_FLUX": args.catalog_mask_min_flux,
        "FPS_MIN_UNMASKED_ANNULUS_FRACTION": (
            args.min_unmasked_annulus_fraction
        ),
        "FPS_MIN_UNMASKED_OFF_COUNTS": args.min_unmasked_off_counts,
    }
    for name, value in optional_settings.items():
        if value is not None:
            run_env[name] = str(value)
    run_env["FPS_ROI_RA"] = str(roi_info["ra_degrees"])
    run_env["FPS_ROI_DEC"] = str(roi_info["dec_degrees"])
    if "analysis_radius_degrees" not in roi_info:
        raise ValueError(
            f"{roi_dir}: query_info.json lacks analysis_radius_degrees"
        )
    run_env["FPS_ROI_RADIUS_DEG"] = str(
        roi_info["analysis_radius_degrees"]
    )
    run_env["FPS_EVENTS_FILE"] = str(roi_dir / "events.txt")
    run_env["FPS_OUTPUT_LABEL"] = relative_output_label(output_dir)
    centers_file = roi_info.get("ownership_centers_file")
    owner_tile = roi_info.get("ownership_tile_id")
    if centers_file or owner_tile:
        if not centers_file or not owner_tile:
            raise ValueError(
                f"{roi_dir}: incomplete unique-ownership metadata"
            )
        run_env["FPS_OWNERSHIP_CENTERS"] = str(project_path(centers_file))
        run_env["FPS_OWNERSHIP_TILE_ID"] = str(owner_tile)
    null_time_model = roi_info.get("null_time_model_file")
    if null_time_model:
        run_env["FPS_NULL_TIME_MODEL"] = str(project_path(null_time_model))
    if args.save_pre_tube:
        run_env["FPS_SAVE_PRE_TUBE"] = "1"
    if args.save_bin_candidates or args.stationary_validation:
        run_env["FPS_SAVE_BIN_CANDIDATES"] = "1"
    # catalogue targets survive the moving-source veto;
    # the catalogue annulus mask retains its production default. ---
    if args.stationary_validation:
        run_env["FPS_STATIONARY_VALIDATION"] = "1"
        run_env["FPS_CATALOG_ON_REGION_VETO"] = "0"

    if args.catalog_fits:
        run_env["FPS_CATALOG_FITS"] = str(project_path(args.catalog_fits))
    if args.scramble_times:
        run_env["FPS_SCRAMBLE_TIMES"] = "1"
        run_env["FPS_SCRAMBLE_SEED"] = str(args.scramble_seed)
    if args.disable_motion_filter:
        run_env["FPS_DISABLE_MOTION_FILTER"] = "1"
    if args.disable_annulus_mask:
        run_env["FPS_CATALOG_ANNULUS_MASK"] = "0"
    return run_env


def file_signature(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def content_signature(path):
    path = Path(path).resolve()
    cache_key = str(path)
    if cache_key in _CONTENT_SIGNATURE_CACHE:
        return _CONTENT_SIGNATURE_CACHE[cache_key]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    signature = {"path": str(path), "size": path.stat().st_size, "sha256": digest}
    _CONTENT_SIGNATURE_CACHE[cache_key] = signature
    return signature


def run_fingerprint(roi_dir, run_env):
    events_manifest = roi_dir / "events.txt"
    event_files = []
    for entry in events_manifest.read_text().splitlines():
        if not entry.strip():
            continue
        event_path = Path(entry.strip())
        if not event_path.is_absolute():
            event_path = PROJECT_ROOT / event_path
        event_files.append(file_signature(event_path))
    catalog = run_env.get("FPS_CATALOG_FITS")
    ownership_centers = run_env.get("FPS_OWNERSHIP_CENTERS")
    null_time_model = run_env.get("FPS_NULL_TIME_MODEL")
    payload = {
        "query_info": content_signature(roi_dir / "query_info.json"),
        "events_manifest": content_signature(events_manifest),
        "event_files": event_files,
        "catalog": file_signature(catalog) if catalog else None,
        "ownership_centers": (
            content_signature(ownership_centers) if ownership_centers else None
        ),
        "null_time_model": (
            content_signature(null_time_model) if null_time_model else None
        ),
        "code": [file_signature(path) for path in STAGE_A_CODE_FILES],
        "fps_environment": {
            key: run_env[key]
            for key in sorted(run_env)
            if key.startswith("FPS_") and key != "FPS_OUTPUT_LABEL"
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest(), payload


def run_one_roi(args, roi_dir, output_root):
    roi_info = load_roi_info(roi_dir)
    roi_id = roi_info["label"]
    output_dir = output_root / roi_id
    output_dir.mkdir(parents = True, exist_ok = True)
    candidate_path = output_dir / "candidates.json"
    summary_path = output_dir / "summary.json"
    manifest_path = output_dir / "run_manifest.json"
    log_path = output_dir / "run.log"
    run_env = build_run_environment(args, roi_dir, roi_info, output_dir)
    fingerprint, fingerprint_inputs = run_fingerprint(roi_dir, run_env)

    reuse = False
    if args.skip_existing and candidate_path.exists() and summary_path.exists():
        try:
            existing_manifest = json.loads(manifest_path.read_text())
            reuse = existing_manifest.get("fingerprint") == fingerprint
        except (FileNotFoundError, json.JSONDecodeError):
            reuse = False
        if reuse:
            print(f"{roi_id}: using fingerprint-matched output", flush = True)
        else:
            print(f"{roi_id}: existing output is stale; recomputing", flush = True)

    if not reuse:
        print(
            (
                f"{roi_id}: running moving search "
                f"(RA={roi_info['ra_degrees']:.4f}, "
                f"Dec={roi_info['dec_degrees']:.4f})"
            ),
            flush = True,
        )
        with log_path.open("w") as log_file:
            process = subprocess.run(
                [sys.executable, str(FPS_SCRIPT)],
                cwd = PROJECT_ROOT,
                env = run_env,
                stdout = log_file,
                stderr = subprocess.STDOUT,
                check = False,
            )
        if process.returncode != 0:
            log_lines = log_path.read_text(errors="replace").splitlines()
            tail = "\n".join(log_lines[-40:])
            print(
                f"\nERROR: {roi_id} Stage A failed; tail of {log_path}:\n{tail}",
                flush=True,
            )
            raise RuntimeError(
                f"{roi_id}: Stage A exited {process.returncode}; see {log_path}"
            )
        if not candidate_path.exists() or not summary_path.exists():
            raise RuntimeError(f"{roi_id}: search completed without required outputs")
        manifest_path.write_text(json.dumps({
            "fingerprint": fingerprint,
            "inputs": fingerprint_inputs,
        }, indent=2))

    shutil.copy2(roi_dir / "query_info.json", output_dir / "query_info.json")
    candidates = json.loads(candidate_path.read_text())

    # this used to unconditionally read
    # summary.json, written by the search, so the driver
    # crashed with FileNotFoundError after every ROI. fps_moving_v5_New now
    # writes the file; the read is also guarded so a missing summary can
    # never kill a 50-ROI batch. ---
    if summary_path.exists():
        summary_records = json.loads(summary_path.read_text())
    else:
        summary_records = []


    return roi_info, output_dir, candidates, summary_records


def copy_nonempty(output_dir, nonempty_root, roi_id):
    destination = nonempty_root / roi_id
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(output_dir, destination)


def remove_nonempty_copy(nonempty_root, roi_id):
    destination = nonempty_root / roi_id
    if destination.exists():
        shutil.rmtree(destination)


def main():
    args = parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be at least one")
    roi_root = project_path(args.roi_root)
    output_root = project_path(args.output_root)
    nonempty_root = project_path(args.nonempty_root)
    output_root.mkdir(parents = True, exist_ok = True)
    nonempty_root.mkdir(parents = True, exist_ok = True)

    roi_dirs = sorted(
        roi_dir
        for roi_dir in roi_root.glob("roi_*")
        if (roi_dir / "events.txt").exists()
        and (roi_dir / "query_info.json").exists()
    )
    if args.roi_ids:
        requested = {
            value.strip() for value in args.roi_ids.split(",") if value.strip()
        }
        roi_dirs = [
            roi_dir for roi_dir in roi_dirs if roi_dir.name in requested
        ]
        missing = requested - {roi_dir.name for roi_dir in roi_dirs}
        if missing:
            raise RuntimeError(
                "Requested ROI IDs not found: " + ", ".join(sorted(missing))
            )
    if not roi_dirs:
        raise RuntimeError(f"No ROI folders found under {roi_root}")

    batch_summary = []
    nonempty_summary = []
    def run(roi_dir):
        return run_one_roi(args, roi_dir, output_root)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        roi_results = list(executor.map(run, roi_dirs))

    for roi_info, output_dir, candidates, summary_records in roi_results:
        record = {
            "roi_id": roi_info["label"],
            "ra_degrees": roi_info["ra_degrees"],
            "dec_degrees": roi_info["dec_degrees"],
            "gal_l_degrees": roi_info["gal_l_degrees"],
            "gal_b_degrees": roi_info["gal_b_degrees"],
            "candidate_count": len(candidates),
            "track_count": len(candidates),
            "survey_geometry": roi_info.get("survey_geometry"),
            "sky_coverage_fraction": roi_info.get("sky_coverage_fraction"),
            "ownership_tile_id": roi_info.get("ownership_tile_id"),
            "ownership_tile_count": roi_info.get("ownership_tile_count"),
            "summary_count": len(summary_records),
            "stage_a_summary": (
                summary_records[0] if len(summary_records) == 1 else None
            ),
            "output_dir": (
                str(output_dir.relative_to(PROJECT_ROOT))
                if output_dir.is_relative_to(PROJECT_ROOT)
                else str(output_dir.resolve())
            ),
        }
        batch_summary.append(record)

        if candidates:
            copy_nonempty(output_dir, nonempty_root, roi_info["label"])
            nonempty_summary.append(record)
            print(
                f"{roi_info['label']}: NONEMPTY ({len(candidates)} moving candidates)",
                flush = True,
            )
        else:
            remove_nonempty_copy(nonempty_root, roi_info["label"])
            print(f"{roi_info['label']}: empty", flush = True)

    batch_summary_path = output_root / "batch_summary.json"
    nonempty_summary_path = nonempty_root / "nonempty_summary.json"
    batch_summary_path.write_text(json.dumps(batch_summary, indent = 2))
    nonempty_summary_path.write_text(json.dumps(nonempty_summary, indent = 2))
    print(
        (
            f"Finished {len(batch_summary)} ROIs; "
            f"{len(nonempty_summary)} nonempty. "
            f"Wrote {batch_summary_path} and {nonempty_summary_path}."
        ),
        flush = True,
    )


if __name__ == "__main__":
    main()
