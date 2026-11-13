import sys
import json
import os
import base64
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


class ScamMarkUnitTests(unittest.TestCase):
    def test_url_analysis_flags_suspicious_tld_and_token(self):
        from scammark.url_analysis import analyze_url

        result = analyze_url("https://secure-login.reward-bonus.example.cfd/path")

        self.assertGreaterEqual(result["risk_score"], 2)
        self.assertIn("suspicious_tld:cfd", result["signals"])
        self.assertIn("suspicious_token:login", result["signals"])
        self.assertIn("suspicious_token:bonus", result["signals"])

    def test_text_analysis_maps_loan_scam(self):
        from scammark.text_analysis import analyze_textual_persuasion

        result = analyze_textual_persuasion(
            title="Fast loan approval",
            text="Bad credit accepted. Guaranteed approval with instant loan payout.",
        )

        self.assertEqual(result["suggested_scam_type"], "Online Loan Scam")
        self.assertGreater(result["risk_score"], 0)
        self.assertTrue(any("loan" in signal for signal in result["signals"]))

    def test_binary_metrics_use_scam_as_positive_class(self):
        from scammark.metrics import compute_binary_metrics

        metrics = compute_binary_metrics(
            y_true=["scam", "scam", "benign", "benign"],
            y_pred=["scam", "benign", "benign", "benign"],
        )

        self.assertAlmostEqual(metrics["accuracy"], 0.75)
        self.assertAlmostEqual(metrics["precision"], 1.0)
        self.assertAlmostEqual(metrics["recall"], 0.5)
        self.assertAlmostEqual(metrics["f1"], 2 / 3)
        self.assertEqual(metrics["confusion"], {"tp": 1, "fp": 0, "tn": 2, "fn": 1})

    def test_pipeline_outputs_required_schema(self):
        from scammark.pipeline import ScamMarkPipeline
        from scammark.model_client import RuleBasedModelClient

        sample = {
            "sample_id": "demo-1",
            "url": "https://loan-bonus.example.cfd",
            "title": "Instant Loan",
            "text": "Bad credit accepted. Guaranteed approval. Contact customer service now.",
            "binary": "scam",
            "scam_type": "Online Loan Scam",
            "image_path": None,
            "mhtml_path": None,
        }
        pipeline = ScamMarkPipeline(model_client=RuleBasedModelClient())
        prediction = pipeline.predict(sample)

        for key in ["binary", "scam_type", "description", "reasoning", "boxes", "evidence_used"]:
            self.assertIn(key, prediction)
        self.assertEqual(prediction["binary"], "scam")
        self.assertEqual(prediction["scam_type"], "Online Loan Scam")
        self.assertIsInstance(prediction["boxes"], list)
        self.assertIn("prompt_stages", prediction["evidence_used"])
        self.assertEqual(
            list(prediction["evidence_used"]["prompt_stages"].keys()),
            ["url", "text", "visual", "cross", "binary", "type", "grounding"],
        )

    def test_prompt_context_excludes_gold_fields(self):
        from scammark.prompts import HIDDEN_SAMPLE_KEYS, build_sample_context, build_stage_prompt

        sample = {
            "sample_id": "demo-1",
            "url": "https://loan-bonus.example.cfd",
            "title": "Instant Loan",
            "visible_text": "Guaranteed approval",
            "binary": "scam",
            "label": "gold label",
            "scam_type": "Online Loan Scam",
            "boxes": [{"xyxy": [1, 2, 3, 4]}],
            "rationale": "gold rationale",
            "truth_description": "gold description",
            "description": "gold page explanation",
            "language": "English",
            "image_path": r"D:\dataset_scam_gold\page.png",
            "mhtml_path": r"D:\dataset_scam_gold\page.mhtml",
            "manual_json_path": r"D:\dataset_scam_gold\page.json",
            "raw_id": "leaky-id",
        }

        context = build_sample_context(sample)
        url_prompt = build_stage_prompt("url", sample)
        text_prompt = build_stage_prompt("text", sample)

        for key in HIDDEN_SAMPLE_KEYS:
            self.assertNotIn(key, context)
            self.assertNotIn(f'"{key}"', url_prompt)
            self.assertNotIn(f'"{key}"', text_prompt)
        self.assertIn("loan-bonus.example.cfd", url_prompt)
        self.assertNotIn("Guaranteed approval", url_prompt)
        self.assertIn("Guaranteed approval", text_prompt)
        self.assertNotIn("gold page explanation", text_prompt)
        self.assertNotIn("dataset_scam_gold", text_prompt)
        self.assertNotIn("leaky-id", text_prompt)

        nested_prompt = build_stage_prompt(
            "visual",
            {
                "url": "https://example.test",
                "title": "Visible title",
                "metadata": {
                    "truth": {"scam_type": "Online Shopping Scam"},
                    "manual_json_path": r"D:\dataset_scam_gold\nested.json",
                },
            },
            {"heuristic": {"visual": {"image_path": r"D:\dataset_scam_gold\nested.png"}}},
        )
        self.assertNotIn("Online Shopping Scam", nested_prompt.split('"allowed_taxonomy_labels"')[0])
        self.assertNotIn("dataset_scam_gold", nested_prompt)

        cascade_prompt = build_stage_prompt(
            "grounding",
            {"url": "https://example.test", "title": "Visible title"},
            {
                "prompt_stages": {
                    "binary": {"binary": "scam", "reasoning": "binary evidence"},
                    "type": {"scam_type": "Online Shopping Scam", "reasoning": "type evidence"},
                }
            },
        )
        self.assertIn('"binary": "scam"', cascade_prompt)
        self.assertIn('"scam_type": "Online Shopping Scam"', cascade_prompt)
        self.assertNotIn("binary evidence", cascade_prompt)
        self.assertNotIn("type evidence", cascade_prompt)

    def test_visual_prompt_covers_dataset_specific_visual_cues(self):
        from scammark.prompts import build_stage_prompt

        prompt = build_stage_prompt(
            "visual",
            {
                "sample_id": "demo-visual",
                "url": "https://example.test",
                "title": "Demo",
                "visible_text": "Telegram support and limited offer",
                "image_path": None,
            },
        )

        for term in [
            "adult",
            "romance",
            "Telegram",
            "QR codes",
            "product grids",
            "crypto dashboards",
            "casino",
            "education",
            "government",
            "bank",
        ]:
            self.assertIn(term, prompt)

    def test_staged_pipeline_uses_separate_prompt_stages(self):
        from scammark.pipeline import ScamMarkPipeline

        class FakeStagedClient:
            def __init__(self):
                self.calls = []

            def complete_json(self, stage, prompt, sample, include_image=False):
                self.calls.append((stage, include_image, prompt))
                responses = {
                    "url": {"risk_level": "high", "evidence_summary": "Suspicious loan domain."},
                    "text": {"candidate_scam_types": ["Online Loan Scam"], "evidence_summary": "Guaranteed approval."},
                    "visual": {"visual_signals": ["login_form"], "evidence_summary": "Loan form."},
                    "cross": {"consistency_findings": ["text_and_visual_align"], "risk_level": "high"},
                    "binary": {"binary": "scam", "confidence": 0.9, "reasoning": "Multiple risky signals."},
                    "type": {
                        "scam_type": "Online Loan Scam",
                        "reasoning": "Loan persuasion dominates.",
                    },
                    "grounding": {
                        "description": "Loan approval page.",
                        "reasoning": "The form and promise are suspicious.",
                        "boxes": [{"xyxy": [10, 20, 200, 120], "reason": "loan form"}],
                    },
                }
                return responses[stage]

            def predict(self, sample, analyses):
                raise AssertionError("staged mode should call complete_json, not one-shot predict")

        sample = {
            "sample_id": "demo-1",
            "url": "https://loan-bonus.example.cfd",
            "title": "Instant Loan",
            "visible_text": "Guaranteed approval. Contact customer service now.",
            "image_path": None,
            "binary": "scam",
            "scam_type": "Online Loan Scam",
        }
        client = FakeStagedClient()
        prediction = ScamMarkPipeline(model_client=client, prompt_mode="staged").predict(sample)

        self.assertEqual([stage for stage, _, _ in client.calls], ["url", "text", "visual", "cross", "binary", "type", "grounding"])
        self.assertTrue(client.calls[2][1])
        self.assertTrue(client.calls[3][1])
        self.assertTrue(client.calls[6][1])
        self.assertEqual(prediction["binary"], "scam")
        self.assertEqual(prediction["scam_type"], "Online Loan Scam")
        self.assertEqual(prediction["boxes"], [{"xyxy": [10, 20, 200, 120], "reason": "loan form"}])
        self.assertIn("prompt_stages", prediction["evidence_used"])

    def test_taxonomy_is_exactly_the_25_observed_gold_classes(self):
        from scammark.taxonomy import CANONICAL_LABELS, INVALID_LABEL, canonicalize_scam_type

        self.assertEqual(len(CANONICAL_LABELS), 25)
        self.assertNotIn("Boss / Friend Impersonation Scam", CANONICAL_LABELS)
        self.assertNotIn("Credit Repair Scam", CANONICAL_LABELS)
        self.assertNotIn("Ticket Refund Scam", CANONICAL_LABELS)
        self.assertEqual(canonicalize_scam_type("Education/Training Scam"), "Education / Training Scam")
        self.assertEqual(
            canonicalize_scam_type("Government/Law Enforcement Impersonation Scam"),
            "Government / Law Enforcement Impersonation Scam",
        )
        self.assertEqual(canonicalize_scam_type("PhisOnline Shopping Scamhing"), "Phishing")
        self.assertEqual(canonicalize_scam_type("Investment Scam"), INVALID_LABEL)
        self.assertEqual(canonicalize_scam_type("fake_online_store"), INVALID_LABEL)

    def test_type_prompt_requires_one_single_label(self):
        from scammark.prompts import build_stage_prompt

        prompt = build_stage_prompt("type", {"sample_id": "one", "title": "demo"})

        self.assertIn("single-label", prompt.lower())
        self.assertIn('"scam_type"', prompt)
        self.assertNotIn("multilabel", prompt.lower())
        self.assertNotIn("primary_scam_type", prompt)
        self.assertNotIn("multilabel_scam_types", prompt)

    def test_staged_pipeline_stops_after_binary_for_benign(self):
        from scammark.pipeline import ScamMarkPipeline

        class BenignClient:
            def __init__(self):
                self.calls = []

            def complete_json(self, stage, prompt, sample, include_image=False):
                self.calls.append(stage)
                if stage == "binary":
                    return {"binary": "benign", "reasoning": "No scam evidence."}
                return {"evidence_summary": "none"}

        client = BenignClient()
        prediction = ScamMarkPipeline(model_client=client, prompt_mode="staged").predict(
            {"sample_id": "benign-1", "url": "https://example.org", "title": "Example"}
        )

        self.assertEqual(client.calls, ["url", "text", "visual", "cross", "binary"])
        self.assertEqual(prediction["binary"], "benign")
        self.assertEqual(prediction["scam_type"], "none")
        self.assertEqual(prediction["boxes"], [])

    def test_force_downstream_runs_type_and_grounding_for_benign(self):
        from scammark.pipeline import ScamMarkPipeline

        class ForcedClient:
            def __init__(self):
                self.calls = []
                self.prompts = {}

            def complete_json(self, stage, prompt, sample, include_image=False):
                self.calls.append(stage)
                self.prompts[stage] = prompt
                if stage == "binary":
                    return {"binary": "benign", "reasoning": "No scam evidence."}
                if stage == "type":
                    return {"scam_type": "Online Shopping Scam", "reasoning": "diagnostic"}
                if stage == "grounding":
                    return {"description": "diagnostic", "reasoning": "diagnostic", "boxes": []}
                return {"evidence_summary": "none"}

        client = ForcedClient()
        prediction = ScamMarkPipeline(
            model_client=client,
            prompt_mode="staged",
            force_downstream=True,
        ).predict({"sample_id": "forced-1", "title": "Example"})

        self.assertEqual(client.calls, ["url", "text", "visual", "cross", "binary", "type", "grounding"])
        self.assertEqual(prediction["binary"], "benign")
        self.assertEqual(prediction["scam_type"], "none")
        self.assertEqual(prediction["evidence_used"]["prompt_stages"]["type"]["scam_type"], "Online Shopping Scam")
        self.assertIn('"force_downstream": true', client.prompts["type"])
        self.assertIn("diagnostic", client.prompts["type"].lower())

    def test_staged_pipeline_does_not_guess_non_schema_binary_labels(self):
        from scammark.pipeline import ScamMarkPipeline

        class InvalidBinaryClient:
            def __init__(self):
                self.calls = []

            def complete_json(self, stage, prompt, sample, include_image=False):
                self.calls.append(stage)
                if stage == "binary":
                    return {"binary": "fraudulent", "reasoning": "Non-schema output."}
                return {"evidence_summary": "none"}

        client = InvalidBinaryClient()
        prediction = ScamMarkPipeline(model_client=client, prompt_mode="staged").predict(
            {"sample_id": "invalid-binary", "title": "Example"}
        )

        self.assertEqual(client.calls, ["url", "text", "visual", "cross", "binary"])
        self.assertEqual(prediction["binary"], "__invalid__")
        self.assertEqual(prediction["scam_type"], "none")
        self.assertEqual(prediction["boxes"], [])

    def test_extract_html_evidence_from_plain_html(self):
        from scammark.evidence_loader import extract_html_evidence

        html = """
        <html><head><title>Bonus Wallet</title></head>
        <body><a href="https://example.cfd/login">Claim bonus</a>
        <button>Verify wallet</button><form><input name="password"></form></body></html>
        """
        evidence = extract_html_evidence(html)

        self.assertEqual(evidence["title"], "Bonus Wallet")
        self.assertIn("Claim bonus", evidence["visible_text"])
        self.assertIn("Verify wallet", evidence["buttons"])
        self.assertIn("password", evidence["forms"])
        self.assertIn("https://example.cfd/login", evidence["links"])

    def test_ai_bbox_normalizes_from_1000_frame(self):
        from scammark.evidence_loader import _normalize_ai_boxes

        with tempfile.TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "demo.png"
            image_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 24)
            boxes = _normalize_ai_boxes([{"bbox": [100, 200, 900, 800], "reason": "demo"}], image_path=None)

        self.assertEqual(boxes[0]["xyxy"], [100.0, 200.0, 900.0, 800.0])
        self.assertEqual(boxes[0]["source"], "ai_json")

    def test_openrouter_default_model_from_env_file(self):
        from scammark.model_client import make_model_client

        with tempfile.TemporaryDirectory() as tmp_dir:
            previous_cwd = Path.cwd()
            old_values = {
                key: os.environ.pop(key, None)
                for key in [
                    "OPENAI_MODEL",
                    "OPENAI_BASE_URL",
                    "OPENROUTER_API_KEY",
                    "SCAMMARK_LLM_MODEL",
                    "SCAMMARK_VLM_MODEL",
                ]
            }
            try:
                tmp_path = Path(tmp_dir)
                (tmp_path / "src").mkdir()
                (tmp_path / "src" / ".env").write_text(
                    "OPENROUTER_API_KEY=test-key\nOPENAI_BASE_URL=https://openrouter.ai/api/v1\n",
                    encoding="utf-8",
                )
                os.chdir(tmp_path)
                client = make_model_client("openai")
            finally:
                os.chdir(previous_cwd)
                for key, value in old_values.items():
                    if value is not None:
                        os.environ[key] = value

        self.assertEqual(client.model, "openai/gpt-4o-mini")

    def test_api_key_normalization_accepts_copied_headers(self):
        from scammark.model_client import _normalize_api_key

        self.assertEqual(_normalize_api_key("Bearer sk-test"), "sk-test")
        self.assertEqual(_normalize_api_key("Authorization: Bearer sk-test"), "sk-test")

    def test_vlm_model_selects_vlm_api_key(self):
        from scammark.model_client import _select_api_config

        old_values = {
            key: os.environ.pop(key, None)
            for key in [
                "SCAMMARK_VLM_MODEL",
                "SCAMMARK_VLM_API_KEY",
                "SCAMMARK_VLM_BASE_URL",
                "SCAMMARK_LLM_API_KEY",
                "OPENAI_API_KEY",
                "OPENAI_BASE_URL",
                "CRAZYROUTER_BASE_URL",
            ]
        }
        try:
            os.environ["SCAMMARK_VLM_MODEL"] = "qwen2.5-vl-7b-instruct"
            os.environ["SCAMMARK_VLM_API_KEY"] = "vlm-key"
            os.environ["SCAMMARK_VLM_BASE_URL"] = "https://cn.crazyrouter.com/v1"
            os.environ["SCAMMARK_LLM_API_KEY"] = "llm-key"
            os.environ["CRAZYROUTER_BASE_URL"] = "https://cn.crazyrouter.com/v1"
            api_key, base_url = _select_api_config("qwen2.5-vl-7b-instruct", "OPENAI_API_KEY", "OPENROUTER_API_KEY", "OPENAI_BASE_URL")
        finally:
            for key, value in old_values.items():
                if value is not None:
                    os.environ[key] = value
                else:
                    os.environ.pop(key, None)

        self.assertEqual(api_key, "vlm-key")
        self.assertEqual(base_url, "https://cn.crazyrouter.com/v1")

    def test_parse_json_response_rejects_empty_content(self):
        from scammark.model_client import _parse_json_response

        with self.assertRaises(json.JSONDecodeError):
            _parse_json_response("")

    def test_parse_json_response_uses_first_balanced_object(self):
        from scammark.model_client import _parse_json_response

        parsed = _parse_json_response('prefix {"binary":"scam","reasoning":"a {quoted} value"} trailing {"extra": true}')

        self.assertEqual(parsed["binary"], "scam")
        self.assertEqual(parsed["reasoning"], "a {quoted} value")

    def test_run_demo_marks_non_schema_binary_invalid_and_gates_downstream(self):
        from scammark.run_demo import normalize_prediction_schema

        self.assertEqual(normalize_prediction_schema({"binary": True, "boxes": []})["binary"], "__invalid__")
        invalid = normalize_prediction_schema(
            {"binary": "garbage", "scam_type": "Online Shopping Scam", "boxes": [{"xyxy": [0, 0, 1, 1]}]}
        )
        self.assertEqual(invalid["binary"], "__invalid__")
        self.assertEqual(invalid["scam_type"], "none")
        self.assertEqual(invalid["boxes"], [])
        self.assertFalse(invalid["schema_valid"])
        self.assertFalse(normalize_prediction_schema(invalid)["schema_valid"])

    def test_demo_prediction_record_keeps_full_gold_truth(self):
        from scammark.run_demo import build_prediction_record

        sample = {
            "sample_id": "gold-1",
            "url": "https://example.test",
            "title": "Example",
            "binary": "scam",
            "scam_type": "Online Shopping Scam",
            "description": "gold page description",
            "boxes": [{"xyxy": [1, 2, 30, 40], "reason": "gold region"}],
            "image_path": "page.png",
            "image_size": [100, 200],
            "label_source": "gold_manual_annotation",
        }
        prediction = {
            "binary": "scam",
            "scam_type": "Online Shopping Scam",
            "description": "prediction",
            "reasoning": "prediction",
            "boxes": [],
            "evidence_used": {"prompt_stages": {}},
        }

        record = build_prediction_record(sample, prediction)

        self.assertEqual(record["truth"]["boxes"], sample["boxes"])
        self.assertEqual(record["truth"]["description"], "gold page description")
        self.assertEqual(record["truth"]["image_size"], [100, 200])
        self.assertEqual(record["input"]["image_path"], "page.png")

    def test_rule_backend_reports_nonempty_text_and_visual_models(self):
        from scammark.model_client import RuleBasedModelClient, selected_model_ids

        identifiers = selected_model_ids(RuleBasedModelClient())

        self.assertEqual(identifiers, {"text_model": "rule-based", "visual_model": "rule-based"})

    def test_local_vlm_transports_image_without_exposing_local_path(self):
        from scammark.model_client import LocalVLMModelClient

        with tempfile.TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "page.png"
            image_path.write_bytes(
                base64.b64decode(
                    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
                )
            )
            client = LocalVLMModelClient(endpoint="http://localhost.test/predict")
            with patch("scammark.model_client._post_local_json", return_value={}) as post:
                client.complete_json(
                    "visual",
                    "prompt",
                    {"url": "https://example.test", "image_path": str(image_path)},
                    include_image=True,
                )

        payload = post.call_args.args[1]
        self.assertTrue(payload["image_url"].startswith("data:image/png;base64,"))
        self.assertNotIn("image_path", payload["sample"])
        self.assertNotIn(str(image_path), json.dumps(payload))

    def test_image_required_stage_fails_when_screenshot_is_missing(self):
        from scammark.model_client import LocalVLMModelClient, _build_api_messages

        client = LocalVLMModelClient(endpoint="http://localhost.test/predict")
        with self.assertRaises(FileNotFoundError):
            client.complete_json("visual", "prompt", {"image_path": "missing.png"}, include_image=True)
        with self.assertRaises(FileNotFoundError):
            _build_api_messages(
                "qwen2.5-vl-7b-instruct",
                {"image_path": "missing.png"},
                "prompt",
                include_image=True,
            )

    def test_taxonomy_maps_only_observed_format_aliases(self):
        from scammark.taxonomy import INVALID_LABEL, canonicalize_scam_type

        self.assertEqual(canonicalize_scam_type("Cryptocurrency Investment Scam"), "Cryptocurrency Investment Scam")
        self.assertEqual(canonicalize_scam_type("phishing"), "Phishing")
        self.assertEqual(canonicalize_scam_type("Education/Training Scam"), "Education / Training Scam")
        self.assertEqual(canonicalize_scam_type("Lottery/Prize Scam"), "Lottery / Prize Scam")
        self.assertEqual(canonicalize_scam_type("Phishing Scam"), INVALID_LABEL)
        self.assertEqual(canonicalize_scam_type("fake_online_store"), INVALID_LABEL)
        self.assertEqual(canonicalize_scam_type("Online_Shopping_Scam"), INVALID_LABEL)
        self.assertEqual(canonicalize_scam_type("Task Based Commission Scam"), INVALID_LABEL)

    def test_retry_repair_preserves_base_truth_and_rejects_stale_truth(self):
        from scammark.repair_run_metrics import merge_retry_predictions

        base_record = {
            "sample_id": "one",
            "input": {"url": "https://example.test"},
            "truth": {
                "binary": "scam",
                "scam_type": "Online Shopping Scam",
                "boxes": [],
                "description": "gold",
            },
            "prediction": {"binary": "__invalid__", "backend_error": True, "boxes": []},
        }
        good_retry = {
            "sample_id": "one",
            "input": {"url": "https://example.test"},
            "prediction": {
                "binary": "scam",
                "scam_type": "Online Shopping Scam",
                "boxes": [],
                "description": "",
                "reasoning": "retry",
                "evidence_used": {"prompt_stages": {}},
            },
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            base = root / "base"
            retry = root / "retry"
            output = root / "out"
            base.mkdir()
            retry.mkdir()
            (base / "metrics.json").write_text(json.dumps({"engineering": {}}), encoding="utf-8")
            (base / "predictions.jsonl").write_text(json.dumps(base_record) + "\n", encoding="utf-8")
            (retry / "predictions.jsonl").write_text(json.dumps(good_retry) + "\n", encoding="utf-8")

            merge_retry_predictions(base, [retry], output)
            repaired = json.loads((output / "predictions.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(repaired["truth"], base_record["truth"])

            stale = dict(good_retry)
            stale["truth"] = {"binary": "benign", "scam_type": "none", "boxes": []}
            (retry / "predictions.jsonl").write_text(json.dumps(stale) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                merge_retry_predictions(base, [retry], output)

    def test_visual_analysis_tolerates_unreadable_image(self):
        from scammark.visual_analysis import analyze_visual_layout

        with tempfile.TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "bad.jpg"
            image_path.write_text("not a real image", encoding="utf-8")
            result = analyze_visual_layout(str(image_path))

        self.assertTrue(result["image_exists"])
        self.assertIsNone(result["width"])
        self.assertIsNone(result["height"])

if __name__ == "__main__":
    unittest.main()
