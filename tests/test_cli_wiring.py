"""Regression checks for the public command-line workflow."""

import unittest
from pathlib import Path

from moving.inject_test_source import format_search_command
from moving.prepare_allsky_v5 import DEFAULT_OUTPUT
from moving.run_fps_moving_v5_rois_New import DEFAULT_ROI_ROOT


class CliWiringTests(unittest.TestCase):
    def test_preparation_and_batch_search_share_default_roi_root(self):
        self.assertEqual(
            Path(DEFAULT_OUTPUT).resolve(),
            Path(DEFAULT_ROI_ROOT).resolve(),
        )

    def test_injection_command_uses_current_required_settings(self):
        command = format_search_command(
            "/tmp/injected/events.txt",
            12.5,
            -6.25,
            18.0,
            "example",
        )
        self.assertIn("FPS_ROI_RADIUS_DEG=18.0", command)
        self.assertIn("FPS_CATALOG_FITS=data/gll_psc_v41.fit", command)
        self.assertNotIn("gll_psc_v35.fit", command)


if __name__ == "__main__":
    unittest.main()
