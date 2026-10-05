#!/usr/bin/env python3
"""Verify release integrity and key cross-product scientific relations."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
IDENTITY = "fl16y_v41_incident_e2_response_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(relative: str):
    return json.loads((ROOT / relative).read_text())


def close(value: float, expected: float, tolerance: float = 1e-10) -> None:
    if not math.isclose(value, expected, rel_tol=tolerance, abs_tol=tolerance):
        raise AssertionError(f"{value!r} != {expected!r}")


def main() -> int:
    manifest = read_json("manifest.json")
    if manifest["analysis_identity"] != IDENTITY:
        raise AssertionError("manifest analysis identity mismatch")
    for relative, expected in manifest["files"].items():
        path = ROOT / relative
        if not path.is_file():
            raise AssertionError(f"missing release file: {relative}")
        actual = sha256(path)
        if actual != expected["sha256"]:
            raise AssertionError(f"hash mismatch: {relative}")
        if path.stat().st_size != expected["bytes"]:
            raise AssertionError(f"size mismatch: {relative}")

    headline = read_json("headline_results.json")
    if headline["analysis_identity"] != IDENTITY:
        raise AssertionError("headline analysis identity mismatch")
    count = headline["real_and_null"]
    if count["observed_candidate_tracks"] != 577:
        raise AssertionError("unexpected real candidate count")
    if len(count["null_counts"]) != 60:
        raise AssertionError("expected 60 randomized-sky campaigns")
    close(sum(count["null_counts"]) / 60, count["null_mean"])
    close(count["plus_one_probability"], 4 / 61)
    close(count["signal_track_upper_limit_90_percent"], 80.75719033562338)

    response = read_json("results/exact_count_response.json")
    if len(response["cells"]) != 45 or response["total_trials"] != 2250:
        raise AssertionError("unexpected injection grid dimensions")
    if sum(cell["recovered"] for cell in response["cells"]) != 725:
        raise AssertionError("injection recovery total mismatch")
    close(
        response["count_upper_limit_recovered_tracks"],
        count["signal_track_upper_limit_90_percent"],
    )
    exact_limits = read_json("results/exact_count_source_limits.json")
    response_nodes = {
        (
            row["velocity_degrees_per_year"],
            row["rate_exact_detected_photons_per_annual_bin"],
        ): row
        for row in response["survey_weighted_nodes"]
    }
    for row in exact_limits["rows"]:
        key = (
            row["velocity_degrees_per_year"],
            row["exact_detected_photons_per_annual_bin"],
        )
        if key not in response_nodes:
            raise AssertionError(f"exact-count response node missing: {key}")
        close(response_nodes[key]["source_limit"], row["source_limit"])
    exact = headline["exact_count"]
    close(
        count["signal_track_upper_limit_90_percent"] / exact["efficiency"],
        exact["source_limit"],
    )

    validation = headline["poisson_validation"]
    if validation["trials"] != 120 or validation["observed_recoveries"] != 50:
        raise AssertionError("unexpected annual-count validation result")
    close(validation["predicted_recoveries"], 53.35596218710311)

    volume = read_json("results/effective_volume.json")
    if len(volume["rows"]) != 9:
        raise AssertionError("unexpected effective-volume grid")
    kernel = read_json("results/abundance_kernel.json")
    if kernel["analysis_identity"] != IDENTITY:
        raise AssertionError("abundance-kernel analysis identity mismatch")

    print(f"PASS: {len(manifest['files'])} files verified for {IDENTITY}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"FAIL: {error}", file=sys.stderr)
        raise
