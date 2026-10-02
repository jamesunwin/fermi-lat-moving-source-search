"""Tests for self-shadow-free adaptive local seeding."""

import unittest

import numpy as np

from moving.adaptive_seeding_v5 import (
    adaptive_density_labels,
    angular_separation_from_zone_representative,
    density_zones,
    poisson_core_threshold,
)


class AdaptiveSeedingTests(unittest.TestCase):
    def test_global_floor_is_never_lowered(self):
        threshold = poisson_core_threshold([0.0, 0.1, 1.0], floor=4)
        self.assertTrue(np.all(threshold >= 4))

    def test_inner_annulus_guard_is_enforced(self):
        with self.assertRaisesRegex(ValueError, "at least 2 epsilon"):
            adaptive_density_labels(
                np.zeros((4, 2)),
                epsilon_radians=0.01,
                inner_radius_scale=1.99,
            )

    def test_compact_source_does_not_raise_its_own_local_threshold(self):
        epsilon = 0.01
        # Background points lie in the 2-5 epsilon annulus.
        angles = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
        background = np.column_stack((
            0.03 * np.sin(angles),
            0.03 * np.cos(angles),
        ))
        base = np.vstack(([[0.0, 0.0]], background))
        _, base_diagnostics = adaptive_density_labels(base, epsilon)

        # Add a bright compact source wholly inside epsilon.  Its photons
        # must not enter the 2-5 epsilon local-background annulus.
        source = np.column_stack((
            0.002 * np.sin(angles),
            0.002 * np.cos(angles),
        ))
        with_source = np.vstack(([[0.0, 0.0]], source, background))
        _, source_diagnostics = adaptive_density_labels(
            with_source, epsilon,
        )
        self.assertEqual(
            int(base_diagnostics["annulus_neighbor_counts"][0]),
            int(source_diagnostics["annulus_neighbor_counts"][0]),
        )
        self.assertEqual(
            int(base_diagnostics["local_min_samples"][0]),
            int(source_diagnostics["local_min_samples"][0]),
        )

    def test_core_components_are_formed_after_local_test(self):
        epsilon = 0.01
        cluster_a = np.array([
            [0.000, 0.000],
            [0.001, 0.000],
            [0.000, 0.001],
            [0.001, 0.001],
        ])
        cluster_b = cluster_a + np.array([0.0, 0.08])
        labels, diagnostics = adaptive_density_labels(
            np.vstack((cluster_a, cluster_b)),
            epsilon,
            minimum_samples_floor=4,
        )
        self.assertEqual(diagnostics["n_core_points"], 8)
        self.assertEqual(set(labels[:4]), {0})
        self.assertEqual(set(labels[4:]), {1})

    def test_density_zones_are_sub_epsilon_and_cover_every_point(self):
        coordinates = np.array([
            [0.0, 0.0],
            [0.001, 0.001],
            [0.101, 0.101],
            [0.102, 0.102],
        ])
        representatives, inverse = density_zones(
            coordinates, zone_size_radians=0.01,
        )
        self.assertEqual(len(inverse), len(coordinates))
        self.assertEqual(len(representatives), 2)
        self.assertEqual(inverse[0], inverse[1])
        self.assertEqual(inverse[2], inverse[3])

    def test_zoned_annulus_expands_inner_guard_by_member_offset(self):
        epsilon = 0.01
        coordinates = np.array([
            [0.000, 0.000],
            [0.004, 0.004],
            [0.030, 0.000],
            [0.000, 0.030],
        ])
        representatives, inverse = density_zones(
            coordinates, zone_size_radians=0.005,
        )
        offsets = angular_separation_from_zone_representative(
            coordinates, representatives, inverse,
        )
        _, diagnostics = adaptive_density_labels(
            coordinates,
            epsilon,
            density_zone_scale=0.5,
        )
        self.assertGreaterEqual(
            diagnostics["effective_inner_radius_scale_minimum"], 2.0,
        )
        self.assertAlmostEqual(
            diagnostics["effective_inner_radius_scale_maximum"],
            2.0 + np.max(offsets) / epsilon,
            places=12,
        )


if __name__ == "__main__":
    unittest.main()
