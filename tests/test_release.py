import subprocess
import sys
import unittest
from pathlib import Path


class ReleaseTest(unittest.TestCase):
    def test_release_verifier(self):
        root = Path(__file__).resolve().parents[1]
        verifier = (
            root
            / "release"
            / "fl16y_v41_incident_e2_response_v1"
            / "verify_release.py"
        )
        result = subprocess.run(
            [sys.executable, str(verifier)],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
