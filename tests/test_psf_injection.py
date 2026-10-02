"""Checks for the instrument-response-backed injection PSF."""

import unittest

import numpy as np

from moving.inject_test_source import (
    KingPsfSampler,
    offset_on_sphere,
    resolve_psf_irf_file,
)
from moving_utils_New import angular_separation_degrees


class KingPsfSamplerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.irf_path = resolve_psf_irf_file()
        except FileNotFoundError as error:
            raise unittest.SkipTest(str(error)) from error
        cls.sampler = KingPsfSampler(cls.irf_path)

    def test_radial_draws_match_tabulated_ideal_cdf(self):
        """One-sample KS check against the interpolated CALDB profile."""
        rng = np.random.default_rng(4127)
        energy_mev = 3000.0
        theta_degrees = 45.0
        conversion_type = 1
        sample = np.sort(np.asarray([
            self.sampler.sample_separation_degrees(
                rng, energy_mev, theta_degrees, conversion_type
            )
            for _ in range(20000)
        ]))
        ideal = self.sampler.radial_cdf(
            sample, energy_mev, theta_degrees, conversion_type
        )
        n_sample = len(sample)
        empirical_upper = np.arange(1, n_sample + 1) / n_sample
        empirical_lower = np.arange(0, n_sample) / n_sample
        statistic = max(
            float(np.max(empirical_upper - ideal)),
            float(np.max(ideal - empirical_lower)),
        )
        # 1% one-sample KS critical value is approximately 1.63/sqrt(n).
        self.assertLess(statistic, 1.63 / np.sqrt(n_sample))

    def test_spherical_offset_has_requested_separation(self):
        ra, dec = offset_on_sphere(359.8, 83.0, 7.0, 1.2)
        measured = float(angular_separation_degrees(
            ra, dec, 359.8, 83.0
        ))
        self.assertAlmostEqual(measured, 7.0, places=10)

    def test_front_and_back_profiles_are_distinct(self):
        radii = np.linspace(0.0, 3.0, 301)
        front = self.sampler.radial_cdf(radii, 1000.0, 40.0, 0)
        back = self.sampler.radial_cdf(radii, 1000.0, 40.0, 1)
        self.assertGreater(float(np.max(np.abs(front - back))), 0.05)


if __name__ == "__main__":
    unittest.main()
