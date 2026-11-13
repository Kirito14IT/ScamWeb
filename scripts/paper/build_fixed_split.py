"""Build the deterministic stratified split used by supervised paper baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


RATES = {"train": 0.70, "validation": 0.15, "test": 0.15}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_rank(seed: int, sample_id: str) -> str:
    return hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).hexdigest()


def allocate_counts(group_sizes: dict[str, int], total: int) -> dict[str, dict[str, int]]:
    allocation: dict[str, dict[str, int]] = {}
    for stratum, size in group_sizes.items():
        train = math.floor(size * RATES["train"])
        validation = math.floor(size * RATES["validation"])
        allocation[stratum] = {
            "train": train,
            "validation": validation,
            "test": size - train - validation,
        }

    targets = {
        "train": round(total * RATES["train"]),
        "validation": round(total * RATES["validation"]),
    }
    targets["test"] = total - targets["train"] - targets["validation"]
    for split in ("train", "validation"):
        needed = targets[split] - sum(counts[split] for counts in allocation.values())
        ranked = sorted(
            group_sizes,
            key=lambda stratum: (
                -(group_sizes[stratum] * RATES[split] - math.floor(group_sizes[stratum] * RATES[split])),
                stratum,
            ),
        )
        cursor = 0
        while needed > 0:
            stratum = ranked[cursor % len(ranked)]
            if allocation[stratum]["test"] > 1:
                allocation[stratum][split] += 1
                allocation[stratum]["test"] -= 1
                needed -= 1
            cursor += 1
    observed = Counter()
    for counts in allocation.values():
        observed.update(counts)
    if dict(observed) != targets:
        raise AssertionError(f"split allocation mismatch: {dict(observed)} != {targets}")
    return allocation


def build_split(records: list[dict[str, Any]], *, seed: int) -> dict[str, Any]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for record in records:
        sample_id = str(record["sample_id"])
        stratum = str(record["scam_type"]) if record["binary"] == "scam" else "benign"
        grouped[stratum].append(sample_id)
    group_sizes = {stratum: len(ids) for stratum, ids in grouped.items()}
    allocation = allocate_counts(group_sizes, len(records))
    sample_to_split: dict[str, str] = {}
    stratum_split_counts: dict[str, dict[str, int]] = {}
    for stratum in sorted(grouped):
        ids = sorted(grouped[stratum], key=lambda sample_id: stable_rank(seed, sample_id))
        counts = allocation[stratum]
        train_end = counts["train"]
        validation_end = train_end + counts["validation"]
        for sample_id in ids[:train_end]:
            sample_to_split[sample_id] = "train"
        for sample_id in ids[train_end:validation_end]:
            sample_to_split[sample_id] = "validation"
        for sample_id in ids[validation_end:]:
            sample_to_split[sample_id] = "test"
        stratum_split_counts[stratum] = dict(Counter(sample_to_split[sample_id] for sample_id in ids))
    split_counts = dict(Counter(sample_to_split.values()))
    if len(sample_to_split) != len(records):
        raise AssertionError("not every manifest sample received exactly one split")
    return {
        "schema_version": "scamweb-v-fixed-split-v1",
        "seed": seed,
        "target_ratios": RATES,
        "sample_count": len(records),
        "split_counts": split_counts,
        "stratification": "benign plus 25 canonical scam subtypes",
        "stratum_split_counts": stratum_split_counts,
        "sample_to_split": dict(sorted(sample_to_split.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()
    records = load_jsonl(args.manifest)
    payload = build_split(records, seed=args.seed)
    payload["manifest_sha256"] = sha256_file(args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: payload[key] for key in ["sample_count", "split_counts", "manifest_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
