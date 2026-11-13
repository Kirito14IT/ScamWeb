"""Integrity checks for the versioned ScamWeb-V frozen input bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


HASHED_FILES = {
    "frozen_manifest": "scamweb_v_manifest.jsonl",
    "frozen_evidence": "web_evidence.jsonl",
    "audit": "mhtml_audit.jsonl",
    "recapture": "mhtml_recapture.jsonl",
    "sample_ids": "sample_ids.txt",
    "boxes": "boxes.jsonl",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _nonempty_line_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def verify_frozen_assets(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    version_path = root / "dataset_version.json"
    if not version_path.is_file():
        raise FileNotFoundError(f"Frozen dataset version file is missing: {version_path}")
    version = json.loads(version_path.read_text(encoding="utf-8"))
    expected_hashes = version.get("sha256")
    if not isinstance(expected_hashes, dict):
        raise ValueError("Frozen dataset version does not contain SHA256 metadata")

    verified: dict[str, str] = {}
    for key, filename in HASHED_FILES.items():
        expected = str(expected_hashes.get(key) or "")
        if not expected:
            continue
        path = root / filename
        if not path.is_file():
            raise FileNotFoundError(f"Frozen asset is missing: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(
                f"{filename} SHA256 mismatch: expected {expected}, observed {actual}"
            )
        verified[filename] = actual

    manifest_path = root / "scamweb_v_manifest.jsonl"
    evidence_path = root / "web_evidence.jsonl"
    expected_count = int(version.get("sample_count") or 0)
    manifest_count = _nonempty_line_count(manifest_path)
    evidence_count = _nonempty_line_count(evidence_path)
    if manifest_count != expected_count or evidence_count != expected_count:
        raise ValueError(
            "Frozen sample counts do not match dataset_version.json: "
            f"manifest={manifest_count}, evidence={evidence_count}, expected={expected_count}"
        )
    return {
        "status": "verified",
        "dataset_version": str(version.get("dataset_version") or ""),
        "sample_count": expected_count,
        "gold_box_count": int(version.get("gold_box_count") or 0),
        "verified_sha256": verified,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify a ScamWeb-V frozen input bundle.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = verify_frozen_assets(args.root)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
