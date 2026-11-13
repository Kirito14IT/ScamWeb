import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from openpyxl import Workbook


class ScamWebDatasetTests(unittest.TestCase):
    def test_read_png_size_reads_only_the_header(self):
        from src.data.scamweb_dataset import read_png_size

        png_header = b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + (321).to_bytes(4, "big") + (654).to_bytes(4, "big")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "large.png"
            path.write_bytes(png_header + (b"x" * 1_000_000))
            with patch.object(Path, "read_bytes", side_effect=AssertionError("full-file read is forbidden")):
                size = read_png_size(path)

        self.assertEqual(size, [321, 654])

    def test_parse_semicolon_boxes(self):
        from src.data.scamweb_dataset import parse_semicolon_boxes

        boxes = parse_semicolon_boxes("10; 20", "30;40", "6; 8", "10; 12")

        self.assertEqual(len(boxes), 2)
        self.assertEqual(boxes[0]["xyxy"], [7.0, 25.0, 13.0, 35.0])
        self.assertEqual(boxes[1]["xyxy"], [16.0, 34.0, 24.0, 46.0])

    def test_parse_mhtml_extracts_page_evidence(self):
        from src.data.scamweb_dataset import parse_mhtml_file

        boundary = "----=_NextPart_Test"
        body = (
            f"Content-Type: multipart/related; boundary=\"{boundary}\"\n\n"
            f"--{boundary}\n"
            "Content-Type: text/html; charset=utf-8\n"
            "Content-Transfer-Encoding: 8bit\n\n"
            "<html><head><title>Loan Approval</title>"
            "<meta name=\"description\" content=\"fast loan\"></head>"
            "<body><a href=\"https://example.com/apply\">Apply</a>"
            "<form><input name=\"phone\" placeholder=\"Phone\"></form>"
            "<button>Get money now</button>"
            "<script>ignored()</script>Visible urgent loan text</body></html>\n"
            f"--{boundary}--\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.mhtml"
            path.write_bytes(body.encode("utf-8"))

            evidence = parse_mhtml_file(path, page_url="https://loan.example.test")

        self.assertEqual(evidence["parse_status"], "ok")
        self.assertEqual(evidence["title"], "Loan Approval")
        self.assertIn("Visible urgent loan text", evidence["visible_text"])
        self.assertEqual(evidence["form_count"], 1)
        self.assertIn("Phone", json.dumps(evidence["forms"], ensure_ascii=False))
        self.assertIn("example.com", evidence["external_domains"])

    def test_parse_mhtml_streams_html_without_reading_all_attachments(self):
        from src.data.scamweb_dataset import parse_mhtml_file

        boundary = "----=_Streaming_Test"
        body = (
            f"Content-Type: multipart/related; boundary=\"{boundary}\"\n\n"
            f"--{boundary}\n"
            "Content-Type: text/html; charset=utf-8\n"
            "Content-Transfer-Encoding: quoted-printable\n\n"
            "<html><body>streamed evidence</body></html>\n"
            f"--{boundary}\n"
            "Content-Type: image/png\n"
            "Content-Transfer-Encoding: base64\n\n"
            + ("A" * 1_000_000)
            + f"\n--{boundary}--\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "large-attachment.mhtml"
            path.write_text(body, encoding="utf-8")
            with patch.object(Path, "read_bytes", side_effect=AssertionError("full-file read is forbidden")):
                evidence = parse_mhtml_file(path)

        self.assertEqual(evidence["parse_status"], "ok")
        self.assertIn("streamed evidence", evidence["visible_text"])

    def test_benign_record_never_keeps_boxes(self):
        from src.data.scamweb_dataset import build_benign_record

        row = {"ID": 1, "URL": "https://www.gov.am", "TITLE": "Government"}
        record = build_benign_record(row, image_path=None, mhtml_path=None, ai_path=None)

        self.assertEqual(record["binary"], "benign")
        self.assertEqual(record["scam_type"], "none")
        self.assertEqual(record["boxes"], [])
        self.assertEqual(record["split"], "all")

    def test_gold_record_uses_json_truth_instead_of_workbook_annotations(self):
        from src.data.scamweb_dataset import build_gold_record

        row = {
            "unit_id": "7",
            "gold_index": 3,
            "URL": "example.test",
            "TITLE": "Workbook title",
            "LABEL": "Wrong workbook label",
            "DESCRIPTION": "Wrong workbook description",
        }
        manual = {
            "label": "Education/Training Scam",
            "description": "Gold JSON page description",
            "language": "English",
            "annotations": [
                {
                    "label": "Education/Training Scam",
                    "description": "manual evidence",
                    "center": {"x": 10, "y": 10},
                    "size": {"width": 4, "height": 4},
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            manual_path = tmp_path / "7.json"
            manual_path.write_text(json.dumps(manual), encoding="utf-8")

            record = build_gold_record(row, manual_path, None, None)

        self.assertTrue(record["is_gold"])
        self.assertEqual(record["scam_type"], "Education / Training Scam")
        self.assertEqual(record["description"], "Gold JSON page description")
        self.assertEqual(record["language"], "English")
        self.assertEqual(record["boxes"][0]["reason"], "manual evidence")

    def test_build_records_enumerates_every_gold_json_stem(self):
        from src.data.scamweb_dataset import build_records

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gold_root = root / "dataset_scam_gold"
            manual_dir = gold_root / "人工标注结果_scam"
            image_dir = gold_root / "截图_scam"
            mhtml_dir = gold_root / "mhtml_scam"
            benign_image_dir = root / "dataset_benign" / "截图_benign"
            benign_mhtml_dir = root / "dataset_benign" / "mhtml_benign"
            for folder in [manual_dir, image_dir, mhtml_dir, benign_image_dir, benign_mhtml_dir]:
                folder.mkdir(parents=True)

            manual = {
                "label": "Online Shopping Scam",
                "description": "Gold description",
                "language": "English",
                "annotations": [
                    {
                        "label": "Online Shopping Scam",
                        "description": "Gold region",
                        "center": {"x": 10, "y": 10},
                        "size": {"width": 4, "height": 6},
                    }
                ],
            }
            png_header = b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + (100).to_bytes(4, "big") + (200).to_bytes(4, "big")
            for stem in ["5685", "5685_2"]:
                (manual_dir / f"{stem}.json").write_text(json.dumps(manual), encoding="utf-8")
                (image_dir / f"{stem}.png").write_bytes(png_header)
                (mhtml_dir / f"{stem}.mhtml").write_text("", encoding="utf-8")
            (benign_image_dir / "1.png").write_bytes(png_header)
            (benign_mhtml_dir / "1.mhtml").write_text("", encoding="utf-8")

            workbook = Workbook()
            gold_sheet = workbook.active
            gold_sheet.title = "scam人工Gold_3416"
            gold_sheet.append(["gold_index", "unit_id", "URL", "TITLE", "LABEL", "DESCRIPTION"])
            gold_sheet.append([1, "5685", "shop-one.test", "Shop one", "Wrong", "Wrong"])
            gold_sheet.append([2, "5685_2", "shop-two.test", "Shop two", "Wrong", "Wrong"])
            benign_sheet = workbook.create_sheet("benign正例总表")
            benign_sheet.append(["ID", "URL", "TITLE"])
            benign_sheet.append([1, "https://gov.example", "Government"])
            workbook.save(root / "总表_0325.xlsx")

            records, _ = build_records(root, root / "missing-old-manifest.jsonl")

        scam_records = [record for record in records if record["binary"] == "scam"]
        self.assertEqual([record["sample_id"] for record in scam_records], ["5685", "5685_2"])
        self.assertEqual([record["url"] for record in scam_records], ["shop-one.test", "shop-two.test"])
        self.assertTrue(all(record["scam_type"] == "Online Shopping Scam" for record in scam_records))
        self.assertTrue(all(record["image_size"] == [100, 200] for record in scam_records))

    def test_frozen_validator_rejects_unpaired_benign_modalities(self):
        from src.data.scamweb_dataset import validate_frozen_dataset

        summary = {
            "total_records": 5658,
            "scam_records": 3416,
            "benign_records": 2242,
            "canonical_subtype_count": 25,
            "gold_box_count": 5489,
            "empty_gold_label_count": 0,
            "invalid_gold_label_count": 0,
            "empty_gold_description_count": 0,
            "empty_gold_region_reason_count": 0,
            "multiple_label_gold_record_count": 0,
            "duplicate_sample_id_count": 0,
            "missing_image_size_count": 0,
            "empty_url_count": 0,
            "empty_title_count": 0,
        }
        inventory = {
            "gold_json_without_workbook_row": 0,
            "gold_json_without_png": 0,
            "gold_json_without_mhtml": 0,
            "benign_files_without_workbook_row": 0,
            "benign_without_png": 1,
            "benign_without_mhtml": 0,
        }

        with self.assertRaises(ValueError):
            validate_frozen_dataset(summary, inventory)

    def test_summary_treats_nonpositive_image_dimensions_as_missing(self):
        from src.data.scamweb_dataset import summarize_records

        record = {
            "sample_id": "benign_1",
            "binary": "benign",
            "scam_type": "none",
            "url": "https://example.test",
            "title": "Example",
            "image_size": [0, 100],
            "boxes": [],
            "is_gold": False,
            "is_weak": False,
            "is_ai_assisted": False,
        }

        summary = summarize_records([record], {})

        self.assertEqual(summary["missing_image_size_count"], 1)


if __name__ == "__main__":
    unittest.main()
