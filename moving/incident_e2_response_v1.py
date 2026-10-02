#!/usr/bin/env python3
"""Incident-E^-2 detected-energy sampling for the FL16Y response recalibration.

This module wraps the shared injection utilities with the energy model for
``fl16y_v41_incident_e2_response_v1``:

    p(E | Omega) proportional to E^-2 E_10(E, Omega), 1--100 GeV.

The ten-year exposure is evaluated at the injected track midpoint using the
nearest CAR pixel and linear exposure interpolation in natural-log energy.
Energy dispersion is not modelled.  The inverse-CDF sampler consumes exactly
one uniform random variate per photon, preserving all unrelated random streams
in paired response checks.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from astropy.io import fits

try:  # Package import.
    from moving import inject_test_source as base_injector
except ImportError:  # Direct execution/import with ``moving`` on sys.path.
    import inject_test_source as base_injector


ANALYSIS_IDENTITY = "fl16y_v41_incident_e2_response_v1"
ENERGY_SAMPLING_MODE = "incident_e2_exposure_folded_v1"
ENERGY_MIN_MEV = 1_000.0
ENERGY_MAX_MEV = 100_000.0
CDF_GRID_POINTS = 32_769
CLUSTERING_BANDS_MEV = (
    (1_000.0, 2_154.4346900318847),
    (2_154.4346900318847, 10_000.0),
    (10_000.0, 21_544.34690031884),
    (21_544.34690031884, 46_415.88833612782),
    (46_415.88833612782, 100_000.0),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, value) -> None:
    """Write JSON atomically so a power loss cannot create a valid partial file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    os.replace(temporary, path)


class ExposureFoldedE2Sampler:
    """Inverse-CDF sampler for incident E^-2 folded through LAT exposure."""

    def __init__(
        self,
        cube_path: Path,
        ra_degrees: float,
        dec_degrees: float,
        *,
        expected_sha256: str | None = None,
        grid_points: int = CDF_GRID_POINTS,
    ):
        self.path = Path(cube_path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(
                f"energy-resolved exposure cube is missing: {self.path}"
            )
        self.sha256 = sha256(self.path)
        if expected_sha256 is not None and self.sha256 != expected_sha256:
            raise RuntimeError(
                "energy-resolved exposure cube hash mismatch: "
                f"{self.sha256} != {expected_sha256}"
            )
        if not np.isfinite(ra_degrees) or not np.isfinite(dec_degrees):
            raise ValueError("exposure direction must be finite")
        if not -90.0 <= float(dec_degrees) <= 90.0:
            raise ValueError("exposure declination lies outside the sky")
        if int(grid_points) < 1025:
            raise ValueError("exposure-folded CDF grid is too coarse")

        with fits.open(self.path, memmap=True) as hdul:
            if "ENERGIES" not in hdul:
                raise ValueError("exposure cube lacks the ENERGIES extension")
            cube = np.asarray(hdul[0].data, dtype=float)
            header = hdul[0].header
            layer_energy = np.asarray(
                hdul["ENERGIES"].data.field(0), dtype=float,
            )
        if cube.ndim != 3:
            raise ValueError("exposure cube must have energy, latitude, longitude axes")
        if not str(header.get("CTYPE1", "")).startswith("RA---CAR"):
            raise ValueError("exposure cube longitude axis is not RA---CAR")
        if not str(header.get("CTYPE2", "")).startswith("DEC--CAR"):
            raise ValueError("exposure cube latitude axis is not DEC--CAR")
        required_wcs = ("CRVAL1", "CRVAL2", "CDELT1", "CDELT2", "CRPIX1", "CRPIX2")
        missing_wcs = [key for key in required_wcs if key not in header]
        if missing_wcs:
            raise ValueError(f"exposure cube lacks WCS fields: {missing_wcs}")
        if len(layer_energy) != cube.shape[0]:
            raise ValueError("exposure energy-axis length mismatch")
        if (
            np.any(~np.isfinite(layer_energy))
            or np.any(np.diff(layer_energy) <= 0.0)
            or layer_energy[0] > ENERGY_MIN_MEV
            or layer_energy[-1] < ENERGY_MAX_MEV
        ):
            raise ValueError("exposure energy grid does not cover 1--100 GeV")

        x = (
            (float(ra_degrees) - float(header["CRVAL1"]))
            / float(header["CDELT1"])
            + float(header["CRPIX1"]) - 1.0
        )
        y = (
            (float(dec_degrees) - float(header["CRVAL2"]))
            / float(header["CDELT2"])
            + float(header["CRPIX2"]) - 1.0
        )
        ix = int(round(x)) % cube.shape[2]
        iy = int(np.clip(round(y), 0, cube.shape[1] - 1))
        layer_exposure = np.asarray(cube[:, iy, ix], dtype=float)
        if np.any(layer_exposure <= 0.0) or np.any(~np.isfinite(layer_exposure)):
            raise ValueError("selected exposure layers are nonpositive or nonfinite")

        log_grid = np.linspace(
            np.log(ENERGY_MIN_MEV), np.log(ENERGY_MAX_MEV), int(grid_points),
        )
        energy_grid = np.exp(log_grid)
        exposure_grid = np.interp(
            log_grid, np.log(layer_energy), layer_exposure,
        )
        density = energy_grid ** -2.0 * exposure_grid
        interval_area = 0.5 * (density[:-1] + density[1:]) * np.diff(energy_grid)
        cumulative = np.concatenate(([0.0], np.cumsum(interval_area)))
        normalization = float(cumulative[-1])
        if not np.isfinite(normalization) or normalization <= 0.0:
            raise ValueError("folded-spectrum normalization is not positive")

        self.ra_degrees = float(ra_degrees) % 360.0
        self.dec_degrees = float(dec_degrees)
        self.pixel = {"ix": ix, "iy": iy}
        self.layer_energy_mev = layer_energy
        self.layer_exposure_cm2_s = layer_exposure
        self.energy_grid_mev = energy_grid
        self.exposure_grid_cm2_s = exposure_grid
        self.density = density
        self.cdf = cumulative / normalization
        self.normalization = normalization

    def sample(
        self,
        rng,
        n,
        gamma,
        e_min=ENERGY_MIN_MEV,
        e_max=ENERGY_MAX_MEV,
    ):
        if float(gamma) != 2.0:
            raise ValueError("incident-E^-2 sampler requires photon index 2")
        if float(e_min) != ENERGY_MIN_MEV or float(e_max) != ENERGY_MAX_MEV:
            raise ValueError("incident-E^-2 sampler is fixed to 1--100 GeV")
        # One draw per photon preserves paired random-number coupling with the
        # base inverse-CDF power-law sampler.
        uniforms = rng.random(int(n))
        return np.interp(uniforms, self.cdf, self.energy_grid_mev)

    def quantile(self, probability: float) -> float:
        if not 0.0 <= float(probability) <= 1.0:
            raise ValueError("probability lies outside [0,1]")
        return float(np.interp(probability, self.cdf, self.energy_grid_mev))

    def probability_between(self, lower_mev: float, upper_mev: float) -> float:
        if not ENERGY_MIN_MEV <= lower_mev < upper_mev <= ENERGY_MAX_MEV:
            raise ValueError("energy interval lies outside 1--100 GeV")
        lower = float(np.interp(lower_mev, self.energy_grid_mev, self.cdf))
        upper = float(np.interp(upper_mev, self.energy_grid_mev, self.cdf))
        return upper - lower

    def band_probabilities(self) -> list[float]:
        return [self.probability_between(lo, hi) for lo, hi in CLUSTERING_BANDS_MEV]

    def summary(self) -> dict:
        return {
            "analysis_identity": ANALYSIS_IDENTITY,
            "mode": ENERGY_SAMPLING_MODE,
            "incident_spectrum": "dN/dE proportional to E^-2",
            "detected_energy_density": "E^-2 times E10(E,Omega)",
            "minimum_mev": ENERGY_MIN_MEV,
            "maximum_mev": ENERGY_MAX_MEV,
            "exposure_cube": {"path": str(self.path), "sha256": self.sha256},
            "exposure_direction": {
                "ra_degrees": self.ra_degrees,
                "dec_degrees": self.dec_degrees,
                "convention": "injected_track_midpoint",
            },
            "nearest_car_pixel": self.pixel,
            "layer_energy_mev": self.layer_energy_mev.tolist(),
            "layer_exposure_cm2_s": self.layer_exposure_cm2_s.tolist(),
            "normalization_arbitrary": self.normalization,
            "energy_interpolation": "linear exposure in natural-log energy",
            "spatial_interpolation": "nearest CAR pixel",
            "cdf_grid_points": len(self.energy_grid_mev),
            "energy_dispersion": "neglected",
            "clustering_band_probabilities": self.band_probabilities(),
            "quantiles_mev": {
                str(q): self.quantile(q) for q in (0.1, 0.25, 0.5, 0.75, 0.9)
            },
        }


def track_midpoint(
    roi_info: dict,
    velocity_degrees_per_year: float,
    position_angle_degrees: float,
    offset_x_degrees: float,
    offset_y_degrees: float,
) -> tuple[float, float]:
    return base_injector.track_position(
        float(roi_info["ra_degrees"]),
        float(roi_info["dec_degrees"]),
        float(velocity_degrees_per_year),
        float(position_angle_degrees),
        0.5 * base_injector.N_BINS,
        float(offset_x_degrees),
        float(offset_y_degrees),
    )


@contextlib.contextmanager
def installed_energy_sampler(sampler: ExposureFoldedE2Sampler):
    """Temporarily install the folded sampler in the shared injector."""
    original = base_injector.sample_power_law
    base_injector.sample_power_law = sampler.sample
    try:
        yield
    finally:
        base_injector.sample_power_law = original


def _rewrite_truth_provenance(overlap: dict, sampler: ExposureFoldedE2Sampler) -> None:
    distribution = sampler.summary()
    truth = overlap["truth"]
    truth["detected_energy_distribution"] = distribution
    truth["spectral_index"] = 2.0
    truth["analysis_identity"] = ANALYSIS_IDENTITY
    for handle in overlap["handles"]:
        handle["truth"] = dict(handle["truth"])
        handle["truth"]["detected_energy_distribution"] = distribution
        handle["truth"]["spectral_index"] = 2.0
        handle["truth"]["analysis_identity"] = ANALYSIS_IDENTITY
        atomic_json(Path(handle["truth_file"]), handle["truth"])
    manifest_path = Path(overlap["handles"][0]["out_dir"]).parent / "overlap_injection.json"
    if manifest_path.parent.exists():
        atomic_json(manifest_path, {
            key: value for key, value in overlap.items() if key != "handles"
        })


def build_incident_e2_overlap_injections(
    roi_root,
    template_roi,
    *,
    exposure_cube,
    expected_exposure_sha256: str | None,
    out_root,
    photons_per_year,
    velocity,
    position_angle,
    seed,
    offset_x,
    offset_y,
    psf_irf_file,
    cap_radius_degrees=None,
    quiet=True,
):
    """Build one fully routed response-folded injection realization."""
    roi_infos = base_injector.load_roi_infos(roi_root)
    if template_roi not in roi_infos:
        raise ValueError(f"unknown template ROI {template_roi}")
    midpoint_ra, midpoint_dec = track_midpoint(
        roi_infos[template_roi], velocity, position_angle, offset_x, offset_y,
    )
    sampler = ExposureFoldedE2Sampler(
        exposure_cube,
        midpoint_ra,
        midpoint_dec,
        expected_sha256=expected_exposure_sha256,
    )
    with installed_energy_sampler(sampler):
        overlap = base_injector.build_overlap_injections(
            roi_root,
            template_roi,
            out_root=out_root,
            photons_per_year=int(photons_per_year),
            velocity=float(velocity),
            position_angle=float(position_angle),
            seed=int(seed),
            offset_x=float(offset_x),
            offset_y=float(offset_y),
            spectral_index=2.0,
            psf_model="king",
            psf_irf_file=str(psf_irf_file),
            cap_radius_degrees=cap_radius_degrees,
            quiet=quiet,
        )
    if not overlap.get("same_physical_photons"):
        raise RuntimeError("injection did not preserve shared photons across caps")
    if not overlap.get("handles"):
        raise RuntimeError("injection trajectory intersects no analysis cap")
    overlap["folded_sampler"] = sampler.summary()
    overlap["analysis_identity"] = ANALYSIS_IDENTITY
    _rewrite_truth_provenance(overlap, sampler)
    return overlap


def validate_full_routing(overlap: dict) -> list[str]:
    """Require one non-empty target handle for every computed membership."""
    memberships = set(overlap.get("memberships", {}))
    handles = {
        handle["query_info"].get("tile_id", handle["query_info"].get("label"))
        for handle in overlap.get("handles", [])
    }
    if not memberships or handles != memberships:
        raise RuntimeError(
            "full-routing accounting mismatch: "
            f"memberships={sorted(memberships)}, handles={sorted(handles)}"
        )
    if overlap.get("cap_radius_source") != "query_info.analysis_radius_degrees":
        raise RuntimeError("production routing did not use per-cap analysis radius")
    return sorted(handles)
