"""Survey geometry and cross-region candidate de-duplication.

This module audits the actual 448-tile survey rather than summing nominal cap
areas. It computes the union of 7.5-degree caps by deterministic Monte Carlo
and applies one identical track matcher to the real search and every stored
scramble campaign.
"""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from moving_utils_New import angular_separation_degrees  # noqa: E402


# Survey-union geometry.
def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def unit_vectors(ra_degrees, dec_degrees):
    ra = np.deg2rad(np.asarray(ra_degrees, dtype=float))
    dec = np.deg2rad(np.asarray(dec_degrees, dtype=float))
    cos_dec = np.cos(dec)
    return np.column_stack((
        cos_dec * np.cos(ra),
        cos_dec * np.sin(ra),
        np.sin(dec),
    ))


def load_roi_centers(roi_root):
    roi_root = Path(roi_root)
    centers = []
    for info_path in sorted(roi_root.glob("roi_*/query_info.json")):
        info = json.loads(info_path.read_text())
        centers.append({
            "roi_id": info.get("tile_id", info.get("label", info_path.parent.name)),
            "ra_degrees": float(info["ra_degrees"]),
            "dec_degrees": float(info["dec_degrees"]),
            "query_info": str(info_path),
            "survey_geometry": info.get("survey_geometry"),
            "ownership_tile_id": info.get("ownership_tile_id"),
            "ownership_centers_file": info.get("ownership_centers_file"),
            "owner_cell_radius_degrees": info.get("owner_cell_radius_degrees"),
        })
    if not centers:
        raise ValueError(f"no ROI query_info.json files under {roi_root}")
    roi_ids = [center["roi_id"] for center in centers]
    if len(set(roi_ids)) != len(roi_ids):
        raise ValueError("ROI identifiers are not unique")
    return centers


def geometry_fingerprint(centers, cap_radius_degrees):
    payload = {
        "cap_radius_degrees": float(cap_radius_degrees),
        "centers": [
            {
                "roi_id": center["roi_id"],
                "ra_degrees": center["ra_degrees"],
                "dec_degrees": center["dec_degrees"],
            }
            for center in centers
        ],
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def compute_union_coverage(
    centers, cap_radius_degrees=7.5, n_samples=1_000_000,
    random_seed=20260727, chunk_size=10_000,
):
    """Uniform-sphere Monte Carlo union area and covered-point multiplicity."""
    if n_samples < 1_000_000:
        raise ValueError("survey coverage requires at least 1,000,000 samples")
    if not 0.0 < cap_radius_degrees < 180.0:
        raise ValueError("cap radius must lie between zero and 180 degrees")
    center_vectors = unit_vectors(
        [center["ra_degrees"] for center in centers],
        [center["dec_degrees"] for center in centers],
    )
    threshold = math.cos(math.radians(cap_radius_degrees))
    rng = np.random.default_rng(random_seed)
    n_covered = 0
    total_memberships = 0
    max_multiplicity = 0
    multiplicity_histogram = {}
    for start in range(0, n_samples, chunk_size):
        size = min(chunk_size, n_samples - start)
        z = rng.uniform(-1.0, 1.0, size)
        longitude = rng.uniform(0.0, 2.0 * math.pi, size)
        radial = np.sqrt(np.maximum(0.0, 1.0 - z * z))
        points = np.column_stack((
            radial * np.cos(longitude),
            radial * np.sin(longitude),
            z,
        ))
        multiplicity = np.sum(points @ center_vectors.T >= threshold, axis=1)
        covered = multiplicity > 0
        n_covered += int(np.count_nonzero(covered))
        total_memberships += int(np.sum(multiplicity[covered]))
        if len(multiplicity):
            max_multiplicity = max(max_multiplicity, int(np.max(multiplicity)))
        values, counts = np.unique(multiplicity, return_counts=True)
        for value, count in zip(values, counts):
            key = str(int(value))
            multiplicity_histogram[key] = (
                multiplicity_histogram.get(key, 0) + int(count)
            )

    fraction = n_covered / n_samples
    standard_error = math.sqrt(
        fraction * (1.0 - fraction) / n_samples
    )
    mean_multiplicity = (
        total_memberships / n_covered if n_covered else 0.0
    )
    cap_fraction = (1.0 - math.cos(math.radians(cap_radius_degrees))) / 2.0
    return {
        "schema_version": 1,
        "method": "uniform_sphere_monte_carlo_union",
        "n_samples": int(n_samples),
        "random_seed": int(random_seed),
        "cap_radius_degrees": float(cap_radius_degrees),
        "n_rois": len(centers),
        "roi_fingerprint_sha256": geometry_fingerprint(
            centers, cap_radius_degrees,
        ),
        "union_sky_fraction": fraction,
        "monte_carlo_standard_error": standard_error,
        "n_uncovered_samples": n_samples - n_covered,
        "all_covered_95cl_lower_bound": (
            0.05 ** (1.0 / n_samples)
            if n_covered == n_samples else None
        ),
        "mean_multiplicity_covered": mean_multiplicity,
        "max_multiplicity_sampled": max_multiplicity,
        "multiplicity_histogram": multiplicity_histogram,
        "summed_cap_area_sky_equivalents": len(centers) * cap_fraction,
        "survey_geometry": sorted(set(
            center.get("survey_geometry") for center in centers
            if center.get("survey_geometry")
        )),
        "note": (
            "The cap union is computed from the actual ROI centres. The "
            "production search uses nearest-centre Voronoi ownership, so "
            "covered points have one accepted owner even where cap "
            "multiplicity exceeds one."
        ),
    }


def cached_union_coverage(
    roi_root, output_path, cap_radius_degrees=7.5,
    n_samples=1_000_000, random_seed=20260727,
):
    centers = load_roi_centers(roi_root)
    fingerprint = geometry_fingerprint(centers, cap_radius_degrees)
    output_path = Path(output_path)
    if output_path.exists():
        cached = json.loads(output_path.read_text())
        if (
            cached.get("roi_fingerprint_sha256") == fingerprint
            and cached.get("n_samples") == n_samples
            and cached.get("random_seed") == random_seed
        ):
            return cached
    result = compute_union_coverage(
        centers,
        cap_radius_degrees=cap_radius_degrees,
        n_samples=n_samples,
        random_seed=random_seed,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))
    return result



# Symmetric real/null cross-region de-duplication.
def candidate_points_by_bin(record):
    return {
        int(point["bin_index"]): point
        for point in record.get("points", [])
        if "bin_index" in point and "ra" in point and "dec" in point
    }


def tracks_match(
    first, second, match_radius_degrees=0.6, min_overlap_bins=3,
    min_fraction_within=1.0,
):
    """Whether two cross-ROI tracks represent the same physical trajectory."""
    if first["_roi_id"] == second["_roi_id"]:
        return False
    first_points = candidate_points_by_bin(first)
    second_points = candidate_points_by_bin(second)
    overlap = sorted(set(first_points) & set(second_points))
    if len(overlap) < min_overlap_bins:
        return False
    separations = np.asarray([
        float(angular_separation_degrees(
            first_points[index]["ra"], first_points[index]["dec"],
            second_points[index]["ra"], second_points[index]["dec"],
        ))
        for index in overlap
    ])
    within_fraction = float(np.mean(separations <= match_radius_degrees))
    return (
        float(np.median(separations)) <= match_radius_degrees
        and within_fraction >= min_fraction_within
    )


def load_tracks(search_root):
    tracks = []
    for candidate_path in sorted(Path(search_root).glob("roi_*/candidates.json")):
        records = json.loads(candidate_path.read_text())
        if not isinstance(records, list):
            raise ValueError(f"malformed candidate file: {candidate_path}")
        for index, record in enumerate(records):
            item = dict(record)
            item["_roi_id"] = candidate_path.parent.name
            item["_candidate_index"] = index
            item["_candidate_id"] = (
                f"{candidate_path.parent.name}_candidate_{index + 1:03d}"
            )
            tracks.append(item)
    return tracks


def deduplicate_tracks(
    tracks, match_radius_degrees=0.6, min_overlap_bins=3,
    min_fraction_within=1.0,
):
    n_tracks = len(tracks)
    parent = list(range(n_tracks))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first, second):
        root_first = find(first)
        root_second = find(second)
        if root_first != root_second:
            parent[root_second] = root_first

    for first_index in range(n_tracks):
        for second_index in range(first_index + 1, n_tracks):
            if tracks_match(
                tracks[first_index], tracks[second_index],
                match_radius_degrees=match_radius_degrees,
                min_overlap_bins=min_overlap_bins,
                min_fraction_within=min_fraction_within,
            ):
                union(first_index, second_index)

    components = {}
    for index in range(n_tracks):
        components.setdefault(find(index), []).append(index)

    def representative_key(index):
        record = tracks[index]
        score = sum(
            float(point.get("sigma", 0.0)) for point in record.get("points", [])
        )
        return (-score, record["_candidate_id"])

    groups = []
    representatives = []
    for indices in sorted(
        components.values(),
        key=lambda values: min(tracks[index]["_candidate_id"] for index in values),
    ):
        representative = min(indices, key=representative_key)
        representatives.append(tracks[representative])
        if len(indices) > 1:
            groups.append({
                "representative": tracks[representative]["_candidate_id"],
                "members": [
                    tracks[index]["_candidate_id"] for index in indices
                ],
            })
    return representatives, groups


def summarize_search(
    search_root, match_radius_degrees=0.6, min_overlap_bins=3,
    min_fraction_within=1.0,
):
    tracks = load_tracks(search_root)
    unique, groups = deduplicate_tracks(
        tracks,
        match_radius_degrees=match_radius_degrees,
        min_overlap_bins=min_overlap_bins,
        min_fraction_within=min_fraction_within,
    )
    return {
        "search_root": str(search_root),
        "n_raw": len(tracks),
        "n_unique": len(unique),
        "n_removed": len(tracks) - len(unique),
        "n_duplicate_groups": len(groups),
        "duplicate_groups": groups,
        "unique_candidate_ids": [
            track["_candidate_id"] for track in unique
        ],
    }


def sample_statistics(values):
    values = [int(value) for value in values]
    mean = sum(values) / len(values)
    variance = (
        sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        if len(values) > 1 else 0.0
    )
    standard_deviation = math.sqrt(variance)
    return {
        "n_campaigns": len(values),
        "mean": mean,
        "standard_deviation": standard_deviation,
        "standard_error": standard_deviation / math.sqrt(len(values)),
    }


def build_deduplication_summary(
    real_root, scramble_root, protocol_path,
    match_radius_degrees=0.6, min_overlap_bins=3,
    min_fraction_within=1.0,
):
    real = summarize_search(
        real_root,
        match_radius_degrees,
        min_overlap_bins,
        min_fraction_within,
    )
    campaigns = []
    seen_seeds = set()
    for seed_dir in sorted(Path(scramble_root).glob("**/seed_*")):
        if not (seed_dir / "batch_summary.json").exists():
            continue
        seed = int(seed_dir.name.split("_", 1)[1])
        if seed in seen_seeds:
            raise ValueError(f"duplicate stored scramble seed {seed}")
        seen_seeds.add(seed)
        summary = summarize_search(
            seed_dir,
            match_radius_degrees,
            min_overlap_bins,
            min_fraction_within,
        )
        summary["seed"] = seed
        summary["role"] = (
            "validation" if "validation" in str(seed_dir.parent)
            else "calibration" if "tuning" in str(seed_dir.parent)
            else "unknown"
        )
        campaigns.append(summary)

    protocol = json.loads(Path(protocol_path).read_text())
    tuning_seeds = {int(seed) for seed in protocol["tuning_seeds"]}
    validation_seeds = {int(seed) for seed in protocol["validation_seeds"]}
    available = {campaign["seed"] for campaign in campaigns}
    missing = sorted((tuning_seeds | validation_seeds) - available)
    if missing:
        raise ValueError(f"missing stored scramble seeds: {missing}")

    subsets = {}
    for name, seeds in (
        ("calibration", tuning_seeds),
        ("validation", validation_seeds),
        ("combined", tuning_seeds | validation_seeds),
    ):
        selected = [campaign for campaign in campaigns if campaign["seed"] in seeds]
        subsets[name] = {
            "seeds": sorted(seeds),
            "raw": sample_statistics([item["n_raw"] for item in selected]),
            "unique": sample_statistics([item["n_unique"] for item in selected]),
            "total_raw": sum(item["n_raw"] for item in selected),
            "total_unique": sum(item["n_unique"] for item in selected),
        }

    return {
        "schema_version": 1,
        "method": "cross_roi_per_bin_track_match",
        "match_radius_degrees": match_radius_degrees,
        "minimum_overlapping_bins": min_overlap_bins,
        "minimum_fraction_within_radius": min_fraction_within,
        "representative_rule": "largest summed per-bin Li-Ma sigma",
        "real": real,
        "scramble_campaigns": sorted(campaigns, key=lambda item: item["seed"]),
        "subsets": subsets,
        "protocol": str(protocol_path),
        "note": (
            "The identical matcher is applied to real and every null campaign. "
            "Production tracks also carry nearest-centre owner_tile_id fields; "
            "the cross-ROI matcher is an independent audit of that ownership."
        ),
    }



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roi-root", default="allsky_queries/data")
    parser.add_argument(
        "--real-root", default="moving/results_full/real_allsky_v1",
    )
    parser.add_argument(
        "--scramble-root",
        default="moving/results_full/calibration/scrambles",
    )
    parser.add_argument(
        "--protocol",
        default="moving/results_full/calibration/null_protocol.json",
    )
    parser.add_argument(
        "--coverage-output",
        default="moving/results_full/calibration/sky_coverage.json",
    )
    parser.add_argument(
        "--dedup-output",
        default="moving/results_full/calibration/deduplication.json",
    )
    parser.add_argument("--cap-radius", type=float, default=7.5)
    parser.add_argument("--samples", type=int, default=1_000_000)
    parser.add_argument("--random-seed", type=int, default=20260727)
    parser.add_argument("--match-radius", type=float, default=0.6)
    parser.add_argument("--min-overlap-bins", type=int, default=3)
    args = parser.parse_args()

    roi_root = project_path(args.roi_root)
    coverage_output = project_path(args.coverage_output)
    coverage = cached_union_coverage(
        roi_root,
        coverage_output,
        cap_radius_degrees=args.cap_radius,
        n_samples=args.samples,
        random_seed=args.random_seed,
    )
    deduplication = build_deduplication_summary(
        project_path(args.real_root),
        project_path(args.scramble_root),
        project_path(args.protocol),
        match_radius_degrees=args.match_radius,
        min_overlap_bins=args.min_overlap_bins,
    )
    dedup_output = project_path(args.dedup_output)
    dedup_output.parent.mkdir(parents=True, exist_ok=True)
    dedup_output.write_text(json.dumps(deduplication, indent=2))

    print(
        f"coverage={coverage['union_sky_fraction']:.8f} +/- "
        f"{coverage['monte_carlo_standard_error']:.3g}; "
        f"mean multiplicity={coverage['mean_multiplicity_covered']:.4f}"
    )
    print(
        f"real candidates: {deduplication['real']['n_raw']} raw -> "
        f"{deduplication['real']['n_unique']} unique"
    )
    validation = deduplication["subsets"]["validation"]
    print(
        "validation null mean: "
        f"{validation['raw']['mean']:.3f} raw -> "
        f"{validation['unique']['mean']:.3f} unique"
    )
    print(f"Wrote {coverage_output}")
    print(f"Wrote {dedup_output}")


if __name__ == "__main__":
    main()
