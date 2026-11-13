from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_balanced_subset(
    manifest_path: Path,
    web_evidence_path: Path,
    output_manifest_path: Path,
    output_web_evidence_path: Path,
    total: int,
) -> dict[str, Any]:
    if total <= 0 or total % 2 != 0:
        raise ValueError("--total must be a positive even number")

    records = load_jsonl(manifest_path)
    evidence_records = load_jsonl(web_evidence_path)
    evidence_by_id = {str(record.get("sample_id") or ""): record for record in evidence_records}

    scam_needed = total // 2
    benign_needed = total // 2
    scam_records = [record for record in records if record.get("binary") == "scam"][:scam_needed]
    benign_records = [record for record in records if record.get("binary") == "benign"][:benign_needed]
    if len(scam_records) < scam_needed or len(benign_records) < benign_needed:
        raise RuntimeError(
            f"Not enough clean records for total={total}: "
            f"scam={len(scam_records)}/{scam_needed}, benign={len(benign_records)}/{benign_needed}"
        )

    subset = scam_records + benign_records
    subset_ids = {str(record.get("sample_id") or "") for record in subset}
    subset_evidence = [record for sid, record in evidence_by_id.items() if sid in subset_ids]

    write_jsonl(output_manifest_path, subset)
    write_jsonl(output_web_evidence_path, subset_evidence)
    return {
        "total": len(subset),
        "scam": len(scam_records),
        "benign": len(benign_records),
        "evidence": len(subset_evidence),
        "manifest": str(output_manifest_path),
        "web_evidence": str(output_web_evidence_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a balanced clean ScamWeb-V subset for demo scaling.")
    parser.add_argument("--manifest", default="outputs/dataset/full_clean_manifest.jsonl")
    parser.add_argument("--web-evidence", default="outputs/dataset/full_clean_web_evidence.jsonl")
    parser.add_argument("--total", type=int, required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--output-web-evidence", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = build_balanced_subset(
        Path(args.manifest),
        Path(args.web_evidence),
        Path(args.output_manifest),
        Path(args.output_web_evidence),
        args.total,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
