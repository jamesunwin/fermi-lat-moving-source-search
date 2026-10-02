"""Spatially adaptive DBSCAN-style seeding for the moving-source search.

Core thresholds use a genuinely local background estimate from a
``2*epsilon`` to ``5*epsilon`` annulus.  The inner exclusion prevents a
source from raising its own threshold.  The global minimum-samples floor is
retained, and connected components are formed only after the per-point core
test.
"""

import math

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import BallTree


def poisson_core_threshold(mu_background, floor):
    """Return ``ceil(mu + 2.5 sqrt(mu) + 1)``, bounded by ``floor``."""
    mu = np.asarray(mu_background, dtype=float)
    if np.any(mu < 0.0) or not np.all(np.isfinite(mu)):
        raise ValueError("background expectations must be finite and nonnegative")
    return np.maximum(
        int(floor),
        np.ceil(mu + 2.5 * np.sqrt(mu) + 1.0).astype(int),
    )


def spherical_disc_area_steradians(radius_radians):
    return 2.0 * math.pi * (1.0 - math.cos(float(radius_radians)))


def spherical_annulus_area_steradians(inner_radians, outer_radians):
    return 2.0 * math.pi * (
        math.cos(float(inner_radians)) - math.cos(float(outer_radians))
    )


def density_zones(coordinates_radians, zone_size_radians):
    """Assign points to sub-epsilon spherical latitude/longitude zones.

    Longitude bin widths are expanded by ``1/cos(latitude)`` so zones retain
    approximately constant angular width.  Each zone is represented by the
    spherical mean of its member points.  Only the background-density query
    is zoned; the core test and connected components still use every photon
    and cross zone boundaries without restriction.
    """
    coordinates = np.asarray(coordinates_radians, dtype=float)
    latitude = coordinates[:, 0]
    longitude = np.mod(coordinates[:, 1], 2.0 * math.pi)
    latitude_index = np.floor(
        (latitude + 0.5 * math.pi) / zone_size_radians
    ).astype(int)
    latitude_center = (
        (latitude_index.astype(float) + 0.5) * zone_size_radians
        - 0.5 * math.pi
    )
    cosine = np.maximum(0.05, np.abs(np.cos(latitude_center)))
    longitude_bins = np.maximum(
        1,
        np.ceil(2.0 * math.pi * cosine / zone_size_radians).astype(int),
    )
    longitude_index = np.floor(
        longitude / (2.0 * math.pi) * longitude_bins
    ).astype(int)
    keys = np.column_stack((latitude_index, longitude_bins, longitude_index))
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    n_zones = int(np.max(inverse)) + 1

    cos_latitude = np.cos(latitude)
    vectors = np.column_stack((
        cos_latitude * np.cos(longitude),
        cos_latitude * np.sin(longitude),
        np.sin(latitude),
    ))
    sums = np.zeros((n_zones, 3), dtype=float)
    np.add.at(sums, inverse, vectors)
    norms = np.linalg.norm(sums, axis=1)
    representatives = sums / norms[:, None]
    representative_latitude = np.arcsin(
        np.clip(representatives[:, 2], -1.0, 1.0)
    )
    representative_longitude = np.mod(
        np.arctan2(representatives[:, 1], representatives[:, 0]),
        2.0 * math.pi,
    )
    return (
        np.column_stack((
            representative_latitude, representative_longitude,
        )),
        inverse,
    )


def angular_separation_from_zone_representative(
    coordinates_radians, representatives_radians, zone_inverse,
):
    """Return each point's great-circle offset from its zone representative."""
    coordinates = np.asarray(coordinates_radians, dtype=float)
    representatives = np.asarray(representatives_radians, dtype=float)
    selected = representatives[np.asarray(zone_inverse, dtype=int)]
    delta_latitude = coordinates[:, 0] - selected[:, 0]
    delta_longitude = coordinates[:, 1] - selected[:, 1]
    haversine = (
        np.sin(0.5 * delta_latitude) ** 2
        + np.cos(coordinates[:, 0])
        * np.cos(selected[:, 0])
        * np.sin(0.5 * delta_longitude) ** 2
    )
    return 2.0 * np.arcsin(np.sqrt(np.clip(haversine, 0.0, 1.0)))


def adaptive_density_labels(
    coordinates_radians,
    epsilon_radians,
    minimum_samples_floor=4,
    inner_radius_scale=2.0,
    outer_radius_scale=5.0,
    density_zone_scale=0.5,
):
    """Return DBSCAN-like labels using a local core threshold per point.

    Parameters
    ----------
    coordinates_radians
        ``(N, 2)`` latitude/longitude coordinates in radians.
    epsilon_radians
        Clustering neighbourhood radius.
    minimum_samples_floor
        Global floor retained at every position.
    inner_radius_scale, outer_radius_scale
        Local-background annulus in units of epsilon.  The inner radius must
        be at least two epsilon to avoid source self-shadowing.

    Notes
    -----
    Core points are those whose epsilon-neighbour count meets their own
    Poisson threshold.  Connected components are then formed among core
    points.  Non-core points within epsilon of a core are assigned to the
    nearest core component as border points.
    """
    coordinates = np.asarray(coordinates_radians, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("coordinates must have shape (N, 2)")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("coordinates must be finite")
    if epsilon_radians <= 0.0:
        raise ValueError("epsilon must be positive")
    if minimum_samples_floor < 1:
        raise ValueError("minimum_samples_floor must be positive")
    if inner_radius_scale < 2.0:
        raise ValueError("inner radius must be at least 2 epsilon")
    if outer_radius_scale <= inner_radius_scale:
        raise ValueError("outer radius must exceed inner radius")
    if not 0.0 < density_zone_scale <= 1.0:
        raise ValueError("density_zone_scale must lie in (0, 1]")
    n_points = len(coordinates)
    if n_points == 0:
        return np.empty(0, dtype=int), {
            "local_mu_background": np.empty(0, dtype=float),
            "local_min_samples": np.empty(0, dtype=int),
            "epsilon_neighbor_counts": np.empty(0, dtype=int),
            "annulus_neighbor_counts": np.empty(0, dtype=int),
            "n_core_points": 0,
        }

    tree = BallTree(coordinates, metric="haversine")
    epsilon_counts = tree.query_radius(
        coordinates, r=epsilon_radians, count_only=True,
    ).astype(int)
    inner_radius = inner_radius_scale * epsilon_radians
    outer_radius = outer_radius_scale * epsilon_radians
    zone_coordinates, zone_inverse = density_zones(
        coordinates, density_zone_scale * epsilon_radians,
    )
    point_zone_offsets = angular_separation_from_zone_representative(
        coordinates, zone_coordinates, zone_inverse,
    )
    maximum_zone_offsets = np.zeros(len(zone_coordinates), dtype=float)
    np.maximum.at(maximum_zone_offsets, zone_inverse, point_zone_offsets)

    # Density queries are shared within small zones for tractability in the
    # Galactic plane.  Expand each representative-centred inner disc by the
    # largest member offset.  By the spherical triangle inequality, every
    # zone member then has a source-free exclusion of at least
    # ``inner_radius_scale * epsilon`` around itself.
    guarded_inner_radii = inner_radius + maximum_zone_offsets
    if np.any(guarded_inner_radii >= outer_radius):
        raise ValueError(
            "density zones are too large for the requested background annulus"
        )
    inner_counts_by_zone = tree.query_radius(
        zone_coordinates, r=guarded_inner_radii, count_only=True,
    ).astype(int)
    outer_counts_by_zone = tree.query_radius(
        zone_coordinates, r=outer_radius, count_only=True,
    ).astype(int)
    annulus_counts_by_zone = np.maximum(
        0, outer_counts_by_zone - inner_counts_by_zone,
    )
    annulus_counts = annulus_counts_by_zone[zone_inverse]
    area_ratio_by_zone = (
        spherical_disc_area_steradians(epsilon_radians)
        / (
            2.0 * math.pi
            * (
                np.cos(guarded_inner_radii)
                - math.cos(float(outer_radius))
            )
        )
    )
    local_mu_by_zone = (
        annulus_counts_by_zone.astype(float) * area_ratio_by_zone
    )
    local_mu = local_mu_by_zone[zone_inverse]
    local_min_samples = poisson_core_threshold(
        local_mu, minimum_samples_floor,
    )
    core_mask = epsilon_counts >= local_min_samples
    core_indices = np.flatnonzero(core_mask)
    labels = np.full(n_points, -1, dtype=int)
    if len(core_indices):
        core_coordinates = coordinates[core_indices]
        core_labels = DBSCAN(
            eps=epsilon_radians,
            min_samples=1,
            metric="haversine",
        ).fit_predict(core_coordinates)
        labels[core_indices] = core_labels

        border_indices = np.flatnonzero(~core_mask)
        if len(border_indices):
            core_tree = BallTree(core_coordinates, metric="haversine")
            distances, nearest = core_tree.query(
                coordinates[border_indices], k=1,
            )
            attach = distances[:, 0] <= epsilon_radians
            labels[border_indices[attach]] = core_labels[
                nearest[attach, 0]
            ]

    return labels, {
        "local_mu_background": local_mu,
        "local_min_samples": local_min_samples,
        "epsilon_neighbor_counts": epsilon_counts,
        "annulus_neighbor_counts": annulus_counts,
        "n_core_points": int(len(core_indices)),
        "inner_radius_scale": float(inner_radius_scale),
        "outer_radius_scale": float(outer_radius_scale),
        "density_zone_scale": float(density_zone_scale),
        "n_density_zones": int(len(zone_coordinates)),
        "maximum_zone_offset_epsilon": float(
            np.max(maximum_zone_offsets) / epsilon_radians
        ),
        "effective_inner_radius_scale_minimum": float(
            np.min(guarded_inner_radii) / epsilon_radians
        ),
        "effective_inner_radius_scale_maximum": float(
            np.max(guarded_inner_radii) / epsilon_radians
        ),
        "minimum_samples_floor": int(minimum_samples_floor),
    }
