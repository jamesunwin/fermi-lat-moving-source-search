#!/usr/bin/env python3
"""Rebuild the SHA-256 inventory for this compact release."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXCLUDED = {"manifest.json", "build_manifest.py", "verify_release.py"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    files = {}
    for path in sorted(item for item in ROOT.rglob("*") if item.is_file()):
        relative = path.relative_to(ROOT).as_posix()
        if relative in EXCLUDED or "__pycache__" in path.parts:
            continue
        files[relative] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    manifest = {
        "schema_version": 1,
        "analysis_identity": "fl16y_v41_incident_e2_response_v1",
        "software_version": "1.0.0",
        "release_date": "2026-10-05",
        "hash_algorithm": "sha256",
        "files": files,
    }
    temporary = ROOT / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(ROOT / "manifest.json")


if __name__ == "__main__":
    main()
