"""Shared production analysis-region and area definitions.

The prepared 448-tile data are BUFFER data: each tile contains photons in a
spherical cap whose radius is recorded in ``query_info.json``.  That cap is
the analysis region used for the DBSCAN density and the Li--Ma on/off areas.
The nearest-centre Voronoi cell is used only to assign each finished track a
unique owner.  Keeping both roles in this object prevents a buffer radius from
being mistaken for a Voronoi-cell radius.

All on/off areas pass through :func:`estimate_spherical_area`.  It samples
uniformly in solid angle within the requested spherical disc or annulus and
applies the same data-boundary and catalogue-mask predicates used for photon
counts.  Stage A starts with this module's 8,192-point deterministic
low-discrepancy sample and adaptively doubles it until the estimated area
error is at most 20% of the local ``N_off`` Poisson error (maximum 262,144).
"""

from dataclasses import dataclass
import csv
import math
from pathlib import Path

import numpy as np


SQUARE_DEGREES_PER_STERADIAN = (180.0 / math.pi) ** 2
DEFAULT_MC_SAMPLES = 8_192
REGION_CONVENTION = "buffer_cap_analysis_voronoi_track_ownership_v1"


def unit_vectors(ra_degrees, dec_degrees):
    """Convert scalar or array ICRS coordinates to an ``(N, 3)`` array."""
    ra = np.deg2rad(np.atleast_1d(np.asarray(ra_degrees, dtype=float)))
    dec = np.deg2rad(np.atleast_1d(np.asarray(dec_degrees, dtype=float)))
    cos_dec = np.cos(dec)
    return np.column_stack((
        cos_dec * np.cos(ra),
        cos_dec * np.sin(ra),
        np.sin(dec),
    ))


def spherical_ring_area_deg2(inner_degrees, outer_degrees):
    """Exact solid angle of a spherical annulus, expressed in square degrees."""
    inner = math.radians(float(inner_degrees))
    outer = math.radians(float(outer_degrees))
    if not 0.0 <= inner < outer <= math.pi:
        raise ValueError("radii must satisfy 0 <= inner < outer <= 180 degrees")
    return (
        2.0 * math.pi * (math.cos(inner) - math.cos(outer))
        * SQUARE_DEGREES_PER_STERADIAN
    )


def sample_spherical_ring(
    center_ra_degrees, center_dec_degrees, inner_degrees, outer_degrees,
    n_samples=DEFAULT_MC_SAMPLES,
):
    """Deterministic uniform-solid-angle samples in a spherical annulus."""
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    inner = math.radians(float(inner_degrees))
    outer = math.radians(float(outer_degrees))
    if not 0.0 <= inner < outer <= math.pi:
        raise ValueError("radii must satisfy 0 <= inner < outer <= 180 degrees")

    # Uniform in cos(radius), with a golden-angle azimuth.  The half-index
    # offset avoids sampling either radial boundary.
    index = np.arange(n_samples, dtype=float) + 0.5
    u = index / n_samples
    cos_radius = math.cos(inner) - u * (
        math.cos(inner) - math.cos(outer)
    )
    sin_radius = np.sqrt(np.maximum(0.0, 1.0 - cos_radius * cos_radius))
    golden = (math.sqrt(5.0) - 1.0) / 2.0
    azimuth = 2.0 * math.pi * np.mod(index * golden, 1.0)

    center = unit_vectors(
        [center_ra_degrees], [center_dec_degrees],
    )[0]
    ra = math.radians(float(center_ra_degrees))
    dec = math.radians(float(center_dec_degrees))
    east = np.array([-math.sin(ra), math.cos(ra), 0.0])
    north = np.array([
        -math.sin(dec) * math.cos(ra),
        -math.sin(dec) * math.sin(ra),
        math.cos(dec),
    ])
    tangent = (
        np.cos(azimuth)[:, None] * north
        + np.sin(azimuth)[:, None] * east
    )
    return cos_radius[:, None] * center + sin_radius[:, None] * tangent


@dataclass(frozen=True)
class AreaEstimate:
    area_deg2: float
    nominal_area_deg2: float
    retained_fraction: float
    region_retained_fraction: float
    masked_fraction_of_region: float
    n_samples: int
    n_region: int
    n_retained: int


@dataclass
class AnalysisRegion:
    """One explicit BUFFER analysis region plus its Voronoi ownership rule."""

    tile_id: str
    center_ra_degrees: float
    center_dec_degrees: float
    data_radius_degrees: float
    ownership_tile_ids: tuple = ()
    ownership_center_vectors: np.ndarray | None = None
    convention: str = REGION_CONVENTION

    def __post_init__(self):
        if not 0.0 < self.data_radius_degrees <= 180.0:
            raise ValueError("data boundary radius must lie in (0, 180]")
        self._center_vector = unit_vectors(
            [self.center_ra_degrees], [self.center_dec_degrees],
        )[0]
        self._data_cosine = math.cos(math.radians(self.data_radius_degrees))

    @property
    def area_deg2(self):
        # Routed through the common estimator; every sample in this disc
        # satisfies the cap predicate, so this equals the exact cap solid angle.
        return estimate_spherical_area(
            self.center_ra_degrees,
            self.center_dec_degrees,
            0.0,
            self.data_radius_degrees,
            self,
            n_samples=1,
        ).area_deg2

    def contains_vectors(self, vectors):
        return np.asarray(vectors) @ self._center_vector >= self._data_cosine

    def owner_tile(self, ra_degrees, dec_degrees):
        if self.ownership_center_vectors is None or not self.ownership_tile_ids:
            return self.tile_id
        vector = unit_vectors([ra_degrees], [dec_degrees])[0]
        return self.ownership_tile_ids[
            int(np.argmax(self.ownership_center_vectors @ vector))
        ]

    def metadata(self):
        return {
            "convention": self.convention,
            "analysis_region": "spherical data-buffer cap",
            "data_boundary_radius_degrees": self.data_radius_degrees,
            "density_area_deg2": self.area_deg2,
            "candidate_ownership": "nearest-centre spherical Voronoi cell",
            "ownership_tile_id": self.tile_id,
            "ownership_tile_count": len(self.ownership_tile_ids) or None,
        }


def load_ownership_centers(path):
    """Load the one canonical centre list used for track ownership."""
    with Path(path).open(newline="") as handle:
        records = list(csv.DictReader(handle))
    if not records:
        raise ValueError(f"empty ownership-centre file: {path}")
    ids = tuple(record["tile_id"] for record in records)
    vectors = unit_vectors(
        [float(record["ra_degrees"]) for record in records],
        [float(record["dec_degrees"]) for record in records],
    )
    return ids, vectors


def estimate_spherical_area(
    center_ra_degrees,
    center_dec_degrees,
    inner_degrees,
    outer_degrees,
    region,
    mask_center_vectors=None,
    mask_radii_degrees=None,
    n_samples=DEFAULT_MC_SAMPLES,
):
    """Estimate ``annulus ∩ region − masks`` with one shared sampler."""
    samples = sample_spherical_ring(
        center_ra_degrees,
        center_dec_degrees,
        inner_degrees,
        outer_degrees,
        n_samples=n_samples,
    )
    in_region = region.contains_vectors(samples)
    retained = in_region.copy()
    mask_vectors = (
        np.asarray(mask_center_vectors, dtype=float)
        if mask_center_vectors is not None else np.empty((0, 3))
    )
    mask_radii = (
        np.asarray(mask_radii_degrees, dtype=float)
        if mask_radii_degrees is not None else np.empty(0)
    )
    if len(mask_vectors):
        if len(mask_vectors) != len(mask_radii):
            raise ValueError("mask centres and radii must have equal length")
        mask_cosines = np.cos(np.deg2rad(mask_radii))
        retained &= ~np.any(
            samples @ mask_vectors.T >= mask_cosines[None, :], axis=1,
        )

    n_region = int(np.count_nonzero(in_region))
    n_retained = int(np.count_nonzero(retained))
    nominal = spherical_ring_area_deg2(inner_degrees, outer_degrees)
    region_fraction = n_region / n_samples
    retained_fraction = n_retained / n_samples
    masked_fraction = (
        (n_region - n_retained) / n_region if n_region else 0.0
    )
    return AreaEstimate(
        area_deg2=nominal * retained_fraction,
        nominal_area_deg2=nominal,
        retained_fraction=retained_fraction,
        region_retained_fraction=region_fraction,
        masked_fraction_of_region=masked_fraction,
        n_samples=int(n_samples),
        n_region=n_region,
        n_retained=n_retained,
    )


def unmasked_photon_count(
    photon_vectors, candidate_vector, inner_degrees, outer_degrees,
    mask_center_vectors=None, mask_radii_degrees=None,
):
    """Count photons in the same annulus-minus-mask region used for area."""
    vectors = np.asarray(photon_vectors, dtype=float)
    candidate = np.asarray(candidate_vector, dtype=float)
    cosine = vectors @ candidate
    inside = (
        cosine <= math.cos(math.radians(inner_degrees))
    ) & (
        cosine >= math.cos(math.radians(outer_degrees))
    )
    mask_vectors = (
        np.asarray(mask_center_vectors, dtype=float)
        if mask_center_vectors is not None else np.empty((0, 3))
    )
    mask_radii = (
        np.asarray(mask_radii_degrees, dtype=float)
        if mask_radii_degrees is not None else np.empty(0)
    )
    if len(mask_vectors):
        mask_cosines = np.cos(np.deg2rad(mask_radii))
        inside &= ~np.any(
            vectors @ mask_vectors.T >= mask_cosines[None, :], axis=1,
        )
    return int(np.count_nonzero(inside))
