"""Unit tests for the production region and annulus implementation."""

import math
import unittest

import numpy as np

from moving.analysis_region_v5 import (
    AnalysisRegion,
    estimate_spherical_area,
    sample_spherical_ring,
    unmasked_photon_count,
    unit_vectors,
)


class AnalysisRegionTests(unittest.TestCase):
    def setUp(self):
        self.region = AnalysisRegion("test", 0.0, 0.0, 18.0)

    def test_unclipped_annulus_matches_analytic_area(self):
        eps = 0.589
        estimate = estimate_spherical_area(
            0.0, 0.0, 2.0 * eps, 5.0 * eps, self.region,
            n_samples=65_536,
        )
        planar = math.pi * (25.0 - 4.0) * eps ** 2
        relative_residual = abs(estimate.area_deg2 / planar - 1.0)
        # The residual here is spherical-vs-planar curvature, not MC error:
        # every low-discrepancy sample is retained far from the boundary.
        self.assertLess(relative_residual, 1.0e-3)
        self.assertEqual(estimate.n_retained, estimate.n_samples)

    def test_masked_area_and_uniform_photons_change_consistently(self):
        eps = 0.589
        n = 262_144
        photon_vectors = sample_spherical_ring(
            0.0, 0.0, 2.0 * eps, 5.0 * eps, n_samples=n,
        )
        # Rotate the uniform photon sample about the candidate direction so
        # the count check is independent of the area sampler's azimuth phase.
        angle = 0.137
        old_y = photon_vectors[:, 1].copy()
        old_z = photon_vectors[:, 2].copy()
        photon_vectors[:, 1] = (
            old_y * math.cos(angle) - old_z * math.sin(angle)
        )
        photon_vectors[:, 2] = (
            old_y * math.sin(angle) + old_z * math.cos(angle)
        )
        mask_vectors = unit_vectors([2.8 * eps], [0.0])
        mask_radii = np.asarray([0.7 * eps])
        estimate = estimate_spherical_area(
            0.0, 0.0, 2.0 * eps, 5.0 * eps, self.region,
            mask_center_vectors=mask_vectors,
            mask_radii_degrees=mask_radii,
            n_samples=n,
        )
        unmasked = unmasked_photon_count(
            photon_vectors,
            unit_vectors([0.0], [0.0])[0],
            2.0 * eps,
            5.0 * eps,
            mask_center_vectors=mask_vectors,
            mask_radii_degrees=mask_radii,
        )
        count_fraction = unmasked / n
        self.assertLess(
            abs(count_fraction - estimate.retained_fraction), 5.0e-4,
        )
        self.assertGreater(estimate.masked_fraction_of_region, 0.0)


if __name__ == "__main__":
    unittest.main()
