"""Tile-consistent empirical time bootstrap for full-sky null campaigns."""

from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def unit_vectors(ra_degrees, dec_degrees):
    ra = np.deg2rad(np.asarray(ra_degrees, dtype=float))
    dec = np.deg2rad(np.asarray(dec_degrees, dtype=float))
    cos_dec = np.cos(dec)
    return np.column_stack((
        cos_dec * np.cos(ra), cos_dec * np.sin(ra), np.sin(dec),
    ))


def splitmix64(values):
    values = np.asarray(values, dtype=np.uint64)
    values = values + np.uint64(0x9E3779B97F4A7C15)
    mixed = values.copy()
    mixed = (mixed ^ (mixed >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    mixed = (mixed ^ (mixed >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return mixed ^ (mixed >> np.uint64(31))


def coherent_scrambled_times(
    ra, dec, run_ids, event_ids, seed, model_path,
):
    """Draw local null times reproducibly from stable photon identifiers."""
    model_path = Path(model_path)
    with np.load(model_path, allow_pickle=False) as model:
        time_edges = np.asarray(model["time_edges_met"], dtype=float)
        counts = np.asarray(model["counts"], dtype=np.int64)
        center_vectors = np.asarray(model["center_vectors"], dtype=float)
    if counts.shape != (len(center_vectors), len(time_edges) - 1):
        raise ValueError(f"Invalid coherent null-time model: {model_path}")
    event_vectors = unit_vectors(ra, dec)
    _, owner_indices = cKDTree(center_vectors).query(event_vectors, k=1)
    run_keys = np.asarray(run_ids, dtype=np.uint32).astype(np.uint64)
    event_keys = np.asarray(event_ids, dtype=np.uint32).astype(np.uint64)
    seed_key = np.uint64(
        (int(seed) * 0xD2B74407B1CE6E93) & ((1 << 64) - 1)
    )
    base = (run_keys << np.uint64(32)) ^ event_keys ^ seed_key
    uniform = (splitmix64(base) >> np.uint64(11)).astype(float) / float(1 << 53)
    fraction = (
        (splitmix64(base ^ np.uint64(0xCA5A826395121157)) >> np.uint64(11))
        .astype(float) / float(1 << 53)
    )
    global_counts = counts.sum(axis=0)
    if global_counts.sum() == 0:
        raise ValueError("Coherent null-time model contains no events")
    scrambled = np.empty(len(event_vectors), dtype=float)
    for owner in np.unique(owner_indices):
        mask = owner_indices == owner
        local_counts = counts[int(owner)]
        if local_counts.sum() == 0:
            local_counts = global_counts
        cumulative = np.cumsum(local_counts, dtype=np.int64)
        targets = uniform[mask] * cumulative[-1]
        bins = np.searchsorted(cumulative, targets, side="right")
        bins = np.clip(bins, 0, len(time_edges) - 2)
        scrambled[mask] = (
            time_edges[bins]
            + fraction[mask] * (time_edges[bins + 1] - time_edges[bins])
        )
    return scrambled
