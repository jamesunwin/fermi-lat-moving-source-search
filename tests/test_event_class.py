import unittest

import numpy as np
from astropy.io import fits

from moving.prepare_allsky_v5 import assign_table_rows
from moving_utils_New import event_class_pass


class EventClassDecodingTests(unittest.TestCase):
    def test_integer_bitmask(self):
        values = np.asarray([0, 128, 129], dtype=np.int32)
        self.assertEqual(
            event_class_pass(values, 128).tolist(), [False, True, True]
        )

    def test_fits_x_msb_first(self):
        values = np.zeros((3, 32), dtype=bool)
        values[1:, 24] = True
        self.assertEqual(
            event_class_pass(values, 128).tolist(), [False, True, True]
        )

    def test_packed_bytes_match_fits_x(self):
        unpacked = np.zeros((3, 32), dtype=bool)
        unpacked[1:, 24] = True
        packed = np.packbits(unpacked, axis=1, bitorder="big")
        np.testing.assert_array_equal(
            event_class_pass(packed, 128),
            event_class_pass(unpacked, 128),
        )

    def test_schema_preserving_assignment_unpacks_x_columns(self):
        unpacked = np.zeros((2, 32), dtype=bool)
        unpacked[:, 24] = True
        packed = np.packbits(unpacked, axis=1, bitorder="big")
        rows = np.empty(2, dtype=[("EVENT_CLASS", "u1", (4,))])
        rows["EVENT_CLASS"] = packed
        output = fits.BinTableHDU.from_columns([
            fits.Column(name="EVENT_CLASS", format="32X"),
        ], nrows=2)
        assign_table_rows(output, rows)
        self.assertEqual(output.columns["EVENT_CLASS"].format, "32X")
        np.testing.assert_array_equal(output.data["EVENT_CLASS"], unpacked)


if __name__ == "__main__":
    unittest.main()
