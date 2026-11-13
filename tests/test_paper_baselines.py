from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scammark.baselines import (
    URL_FEATURE_NAMES,
    records_for_split,
    resolve_dataset_asset,
    url_features,
    verify_split_manifest,
)


class PaperBaselineTests(unittest.TestCase):
    def test_url_features_have_stable_shape(self) -> None:
        values = url_features("https://secure-login.example.test:8443/pay?id=42")
        self.assertEqual(len(values), len(URL_FEATURE_NAMES))
        self.assertTrue(all(isinstance(value, float) for value in values))

    def test_url_features_tolerate_malformed_port(self) -> None:
        values = url_features("example.test:not-a-port/login")
        self.assertEqual(values[URL_FEATURE_NAMES.index("has_explicit_port")], 1.0)

    def test_records_for_split_enforces_frozen_count(self) -> None:
        records = [{"sample_id": "a"}, {"sample_id": "b"}]
        split = {
            "sample_to_split": {"a": "train", "b": "test"},
            "split_counts": {"train": 1, "test": 1},
        }
        self.assertEqual(records_for_split(records, split, "test"), [{"sample_id": "b"}])

    def test_manifest_hash_is_checked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.jsonl"
            manifest.write_text(json.dumps({"sample_id": "a"}) + "\n", encoding="utf-8")
            digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
            verify_split_manifest({"manifest_sha256": digest}, manifest)
            with self.assertRaises(ValueError):
                verify_split_manifest({"manifest_sha256": "0" * 64}, manifest)

    def test_dataset_asset_remapping(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "scamweb_dataset"
            target = root / "dataset_benign" / "screenshots" / "1.png"
            target.parent.mkdir(parents=True)
            target.write_bytes(b"png")
            source = r"D:\\datasets\\scamweb_dataset\\dataset_benign\\screenshots\\1.png"
            self.assertEqual(resolve_dataset_asset(source, root), target)


if __name__ == "__main__":
    unittest.main()
