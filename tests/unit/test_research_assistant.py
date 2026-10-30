import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.runtime import research_assistant as research
from src.runtime.runner import ArtifactState
from src.runtime.skills import Registry, Skill, load_body
from src.runtime.skill_bundle import bundle_files, render_references
from src.runtime.pi_bridge import _snapshot_skills
from src.governance import release
from src.evolution.session_feedback import record_feedback, export_development
from src.evolution.crystallize import propose_skill_from_failures
from src.eval.open_research import assess_annotations, validate_group_partition
from src.retrieval.chunking import atomic_blocks, assemble
from src.util import read_json, sha256_file, sha256_obj


class ResearchAssistantTest(unittest.TestCase):
    def request(self):
        return research.request_from_dict({"task_id": "task-one", "task_type": "open_research",
            "query": "What does the assay show?", "project_id": "repair-test", "session_id": "session-one",
            "evidence": [{"source_id": "paper-one", "text": "The assay has no control group.",
                          "url": "https://example.org/paper", "locator": "Methods p.2"}]})

    def state(self):
        return ArtifactState("open-test", "Answer from evidence.", Registry("open-test", []))

    def test_open_answer_has_text_and_full_source_provenance_without_gold(self):
        request, state, cfg = self.request(), self.state(), {"runtime": {"provider": "fake"}}
        plan = research.build_plan(request, cfg, state)
        result = research.execute(request, cfg, plan["plan_sha256"], state=state)
        response = result["response"]
        self.assertTrue(response["answer"])
        self.assertEqual(response["answer_index"], -1)
        self.assertEqual(response["citations"][0]["content_sha256"],
                         plan["sources"][0]["content_sha256"])
        self.assertEqual(sha256_file(Path(result["run_dir"]) / "raw-model-output.json"),
                         read_json(Path(result["run_dir"]) / "receipt.json")["raw_model_output_sha256"])
        self.assertEqual(response["citations"][0]["locator"], "Methods p.2")
        self.assertNotIn("correct", response["execution_summary"])
        self.assertFalse(response["execution_summary"]["citation_entailment_verified"])
        request.query = "Changed question"
        with self.assertRaises(ValueError):
            research.execute(request, cfg, plan["plan_sha256"], state=state)

    def test_typed_request_rejects_gold_and_missing_or_duplicate_sources(self):
        value = self.request().to_dict()
        for change in ({"correct_index": 0}, {"options": ["yes", "no"]},
                       {"task_type": "mcq"}, {"task_id": "../escape"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                research.request_from_dict({**value, **change})
        value["evidence"] *= 2
        with self.assertRaises(ValueError):
            research.request_from_dict(value)

    def test_open_provider_citations_are_verified_against_visible_source(self):
        request, state = self.request(), self.state()
        cfg = {"runtime": {"provider": "http", "open_research_structured_parse_failures": True}, "models": {"task": {"provider": "http"}}}
        claims = {"status": "ok", "claims": [{"text": "No control was used.",
            "citations": [{"source_id": "paper-one", "quote": "no control group"}]}],
            "limitations": [{"text": "The assay has no control group, limiting causal inference.",
                "citations": [{"source_id": "paper-one", "quote": "no control group"}]}]}
        prompt = {}
        def chat_measured(system, user, **kwargs):
            prompt["system"] = system
            return SimpleNamespace(content=json.dumps(claims), cost={"usd": None})
        provider = SimpleNamespace(chat_measured=chat_measured)
        plan = research.build_plan(request, cfg, state)
        format_spec = research.effective_config(cfg, plan["limits"])["models"]["task"]["extra_body"]["response_format"]
        self.assertEqual(format_spec["type"], "json_schema")
        self.assertEqual(format_spec["json_schema"]["schema"]["properties"]["limitations"]["items"]["required"],
                         ["text", "citations"])
        result = research.execute(request, cfg, plan["plan_sha256"], state=state, provider=provider)
        self.assertIn("Only include an optional limitation when the user asks", prompt["system"])
        self.assertIn("[paper-one]", result["response"]["answer"])
        self.assertEqual(len(result["response"]["citations"]), 2)
        self.assertIn("limiting causal inference", result["response"]["citations"][1]["claim"])
        claims["limitations"] = ["An uncited empirical limitation."]
        failed = research.execute(request, cfg, plan["plan_sha256"], state=state, provider=provider)
        self.assertEqual(failed["response"]["status"], "error")
        self.assertTrue(failed["response"]["execution_summary"]["parse_failure"])
        claims["limitations"] = [{"text": "An unsupported limitation.",
            "citations": [{"source_id": "paper-one", "quote": "a fabricated quotation"}]}]
        failed = research.execute(request, cfg, plan["plan_sha256"], state=state, provider=provider)
        self.assertEqual(failed["response"]["status"], "error")
        claims["limitations"] = []
        result = research.execute(request, cfg, plan["plan_sha256"], state=state, provider=provider)
        self.assertEqual(len(result["response"]["citations"]), 1)
        self.assertIn("范围提示", result["response"]["answer"])
        fenced_provider = SimpleNamespace(chat_measured=lambda *args, **kw:
            SimpleNamespace(content="```json\n" + json.dumps(claims) + "\n```", cost={"usd": None}))
        self.assertEqual(research.execute(request, cfg, plan["plan_sha256"], state=state,
            provider=fenced_provider)["response"]["status"], "ok")
        with self.assertRaises(ValueError):
            research._parse_answer("```json\n" + json.dumps(claims), [], 8000)
        for bad in ("invented-source", "paper-one"):
            claims["claims"][0]["citations"][0] = {"source_id": bad, "quote": "a fabricated quotation"}
            failed = research.execute(request, cfg, plan["plan_sha256"], state=state, provider=provider)
            self.assertEqual(failed["response"]["status"], "error")
            self.assertTrue(failed["response"]["execution_summary"]["parse_failure"])

    def test_open_provider_repairs_unique_html_whitespace_quote_only(self):
        visible = [{"source_id": "paper-one",
            "text": "<h4>Results</h4> Participants improved (p < 0.001).",
            "visible_text": "<h4>Results</h4> Participants improved (p < 0.001).",
            "url": "", "locator": "", "retrieved_at": "", "content_sha256": "fixture"}]
        payload = {"status": "ok", "claims": [{"text": "Participants improved.",
            "citations": [{"source_id": "paper-one", "quote": "Participants improved (p < 0.001)."}]}],
            "limitations": []}
        status, answer, citations = research._parse_answer(json.dumps(payload), visible, 8000)
        self.assertEqual(status, "ok")
        self.assertEqual(citations[0].evidence_excerpt,
                         "</h4> Participants improved (p < 0.001).")

    def test_open_provider_allows_distinct_quotes_from_one_source(self):
        visible = [{"source_id": "paper-one", "text": "First finding. Second finding.",
            "visible_text": "First finding. Second finding.", "url": "", "locator": "",
            "retrieved_at": "", "content_sha256": "fixture"}]
        payload = {"status": "ok", "claims": [{"text": "Two findings.", "citations": [
            {"source_id": "paper-one", "quote": "First finding."},
            {"source_id": "paper-one", "quote": "Second finding."}]}], "limitations": []}
        _, _, citations = research._parse_answer(json.dumps(payload), visible, 8000)
        self.assertEqual([item.evidence_excerpt for item in citations],
                         ["First finding.", "Second finding."])
        payload["claims"][0]["citations"].append({"source_id": "paper-one", "quote": "First finding."})
        with self.assertRaises(ValueError):
            research._parse_answer(json.dumps(payload), visible, 8000)

    def test_open_provider_defaults_optional_limitations_to_empty(self):
        visible = [{"source_id": "paper-one", "text": "A reported finding.",
            "visible_text": "A reported finding.", "url": "", "locator": "",
            "retrieved_at": "", "content_sha256": "fixture"}]
        payload = {"status": "ok", "claims": [{"text": "A finding.", "citations": [
            {"source_id": "paper-one", "quote": "A reported finding."}]}]}
        status, _, citations = research._parse_answer(json.dumps(payload), visible, 8000)
        self.assertEqual(status, "ok")
        self.assertEqual(len(citations), 1)

    def test_no_sources_abstains_and_consent_is_required_for_development_fuel(self):
        request, cfg, state = self.request(), {"runtime": {"provider": "fake"}}, self.state()
        result = research.execute(request, cfg, research.build_plan(request, cfg, state)["plan_sha256"], state=state)
        root = Path(result["run_dir"])
        record_feedback(root, category="procedure", correction="Distinguish controls and technical replicates.",
                        reviewer="human-one")
        approved = record_feedback(root, category="procedure", correction="Check biological replication before inference.",
                        reviewer="human-one", use_for_evolution=True)
        with tempfile.TemporaryDirectory() as directory:
            exported = export_development(request.project_id, Path(directory) / "dev.json")
        self.assertTrue(any(row["feedback_id"] == read_json(approved)["feedback_id"] for row in exported["examples"]))
        self.assertTrue(all(row["split"] == "dev_optimize" for row in exported["examples"]))
        self.assertFalse(exported["heldout_use_allowed"])
        with self.assertRaises(ValueError):
            record_feedback(root, category="preference", correction="Prefer Chinese.", reviewer="human-one",
                            use_for_evolution=True)
        request.evidence = []
        plan = research.build_plan(request, cfg, state)
        empty = research.execute(request, cfg, plan["plan_sha256"], state=state)
        self.assertEqual(empty["response"]["status"], "abstain")

    def test_invalid_http_answer_preserves_raw_output_for_audit(self):
        request, state = self.request(), self.state()
        cfg = {"runtime": {"provider": "http", "open_research_structured_parse_failures": True}, "models": {"task": {"provider": "http"}}}
        plan = research.build_plan(request, cfg, state)
        provider = SimpleNamespace(chat_measured=lambda *args, **kw:
            SimpleNamespace(content="unfinished JSON", cost={"usd": 0.001}))
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            with patch.object(research, "artifacts_path", return_value=run_dir):
                result = research.execute(request, cfg, plan["plan_sha256"], state=state, provider=provider)
            self.assertEqual(result["response"]["status"], "error")
            self.assertTrue(result["response"]["execution_summary"]["parse_failure"])
            raw = run_dir / "raw-model-output.json"
            self.assertEqual(read_json(raw)["content"], "unfinished JSON")
            status = read_json(run_dir / "status.json")
            self.assertEqual(status["raw_model_output_sha256"], sha256_file(raw))
            self.assertEqual(status["request_accounting"]["physical_http_attempts"], 0)
            from src.runtime.research_tools import audit_open_research_trace
            self.assertEqual(audit_open_research_trace(run_dir)["status"], "error")
            original = raw.read_bytes()
            raw.write_bytes(original + b"\n")
            with self.assertRaises(ValueError):
                audit_open_research_trace(run_dir)

    def test_bundle_references_and_scripts_survive_immutable_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "references").mkdir()
            reference = root / "references" / "rubric.md"
            reference.write_text("Check evidence provenance.", encoding="utf-8")
            (root / "scripts").mkdir()
            script = root / "scripts" / "helper.py"
            script.write_text("print('inert dependency')", encoding="utf-8")
            skill = Skill("bundle-test", "G-main", "assay", body="## Procedure\n1. Read.\n2. Compare.\n3. Explain.",
                bundle_root=str(root), permissions=["references.read"], license="MIT",
                bundle_files={"references/rubric.md": sha256_file(reference), "scripts/helper.py": sha256_file(script)})
            registry = Registry("bundle-test", [skill])
            snapshot = release._materialize_bodies(registry, "bundle-repair-test")
            reference.write_text("mutable source edited", encoding="utf-8")
            self.assertIn("Check evidence", render_references(snapshot.skills[0]))
            self.assertEqual(len(bundle_files(snapshot.skills[0])), 2)
            pi = _snapshot_skills(snapshot)[0]
            self.assertTrue((Path(pi["filePath"]).parent / "scripts/helper.py").is_file())
            self.assertIn("Check evidence", pi["content"])
            self.assertNotIn("print('inert", pi["content"])
            altered = copy.deepcopy(snapshot.skills[0])
            altered.bundle_files["../outside"] = "a" * 64
            with self.assertRaises(ValueError):
                bundle_files(altered)

    def test_new_skill_trial_keeps_declared_bundle_and_snapshot_rejects_unsafe_names(self):
        from src.contracts.types import Candidate, content_sha256
        from src.evolution.apply import state_with
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "references").mkdir()
            reference = root / "references/controls.md"
            reference.write_text("Check the control condition.", encoding="utf-8")
            body = "## Procedure\n1. Read.\n2. Check controls.\n3. Cite evidence."
            skill = Skill("new-bundle-check", "G-main", "assay controls", body=body, license="MIT",
                bundle_root=str(root), bundle_files={"references/controls.md": sha256_file(reference)},
                dependencies=["reference text only"], permissions=["references.read"])
            candidate = Candidate("candidate-bundle-test", "open-test", "new_skill", skill.name,
                new_content=body, content_sha256=content_sha256(body), extra={"skill": skill.to_dict()})
            trial = state_with(self.state(), candidate)
            self.assertEqual(trial.registry.get(skill.name).bundle_files, skill.bundle_files)
            snapshot = release._materialize_bodies(trial.registry, "new-bundle-trial-fixture")
            reference.write_text("source changed", encoding="utf-8")
            self.assertIn("Check the control", render_references(snapshot.get(skill.name)))
            candidate.artifact_id = "../escape"
            with self.assertRaises(ValueError):
                state_with(self.state(), candidate)

    def test_chunk_limits_include_omission_markers_and_unbroken_sentences(self):
        text = "unbroken" * 1000 + " important assay measurement.\n" + "irrelevant " * 500
        self.assertTrue(all(len(block) <= 120 for block in atomic_blocks(text, max_chars=120)))
        for budget in (20, 150, 800):
            rendered, _ = assemble(text, {"important", "assay"}, set(), budget_chars=budget)
            self.assertLessEqual(len(rendered), budget)

    def test_live_skill_proposal_is_not_a_template_and_rejects_gate_inputs(self):
        proposal = {"name": "assay-checks", "description": "Assay measurement evidence checks",
            "triggers": ["assay", "measurement"], "body": "## Procedure\n1. Read evidence.\n2. Compare labels.\n3. Explain limits."}
        provider = SimpleNamespace(chat=lambda *args, **kw: json.dumps(proposal))
        with patch("src.evolution.reflection_backend.LlmBackend._format_cases", return_value="visible failures"):
            skill = propose_skill_from_failures(self.state(), [{"split": "dev_optimize"}], {}, provider=provider)
            self.assertEqual(skill.body, proposal["body"])
            with self.assertRaises(ValueError):
                propose_skill_from_failures(self.state(), [{"split": "selection_gate"}], {}, provider=provider)

    def test_open_scoring_requires_independent_human_rubric_and_complete_annotations(self):
        response = {"answer": "An evidence-bound claim", "citations": [{"source_id": "paper-one"}]}
        rubric = {"schema": "open-research-rubric-v1", "author": "author", "authorship": "independent_human",
                  "response_sha256": sha256_obj(response), "items": [{"item_id": "limits"}]}
        rows = [{"reviewer": reviewer, "kind": kind, "item_id": identity, "supported": True,
            "source": "independent_human", "rationale": "Verified against supplied source.",
            "response_sha256": sha256_obj(response), "rubric_sha256": sha256_obj(rubric)}
            for reviewer in ("one", "two") for kind, identity in (("rubric", "limits"), ("citation", "0"))]
        self.assertEqual(assess_annotations(response, rubric, rows)["agreement"], 1)
        with self.assertRaises(ValueError):
            assess_annotations(response, rubric, rows[:-1])
        with self.assertRaises(ValueError):
            validate_group_partition({split: [{"task_id": split, "group_id": "same-article"}]
                for split in ("dev_optimize", "selection_gate", "final_holdout")})

    def test_consented_feedback_creates_a_real_proposal_without_publication(self):
        from src.evolution.feedback_skill import build_feedback_skill_plan, execute_feedback_skill
        request, cfg, state = self.request(), {"runtime": {"provider": "fake"}}, self.state()
        request.project_id = "fuel-adapter-test"
        result = research.execute(request, cfg, research.build_plan(request, cfg, state)["plan_sha256"], state=state)
        record_feedback(Path(result["run_dir"]), category="procedure", correction="Check controls before causal inference.",
                        reviewer="human", use_for_evolution=True,
                        evidence_quotes=[{"source_id": "paper-one", "quote": "no control group"}])
        with tempfile.TemporaryDirectory() as directory:
            dataset = export_development(request.project_id, Path(directory) / "fuel.json")
        proposal = {"name": "assay-control-check", "description": "Check assay controls",
            "triggers": ["assay", "control"], "body": "## Procedure\n1. Read material.\n2. Check controls.\n3. Report limitations."}
        provider = SimpleNamespace(chat=lambda *args, **kw: json.dumps(proposal))
        plan = build_feedback_skill_plan(dataset, cfg, state)
        result = execute_feedback_skill(dataset, cfg, plan["plan_sha256"], state=state, provider=provider)
        self.assertEqual(result["status"], "proposed_pending_independent_evaluation")
        self.assertFalse(result["publication_authorized"])
        dataset["examples"][0]["expected_behavior"] = "tampered"
        dataset["dataset_sha256"] = sha256_obj(dataset["examples"])
        with self.assertRaises(ValueError):
            build_feedback_skill_plan(dataset, cfg, state)

    def test_skill_lifecycle_preserves_archives_and_never_retires_verifiers(self):
        from src.evolution.skill_curation import propose_curation
        from src.runtime.skills import route, select_skill, render_catalog
        body = "## Procedure\n1. Read.\n2. Compare.\n3. Explain."
        registry = Registry("test", [Skill("one", "G-main", "assay measurement", body=body),
            Skill("two", "G-main", "assay interpretation", body=body),
            Skill("verify", "G-verify", "citation verification", body=body, frozen=True)])
        proposal = propose_curation(registry, retire=["one"])
        self.assertFalse(proposal["publication_authorized"])
        self.assertEqual(proposal["trial_bodies"]["one"], body)
        with self.assertRaises(ValueError):
            propose_curation(registry, retire=["verify"])
        registry.skills[0].lifecycle = "retired"
        self.assertNotIn("one", [row["name"] for row in route("assay", registry)])
        with self.assertRaises(ValueError):
            select_skill("assay", registry, pinned_skill="one")
        with self.assertRaises(ValueError):
            render_catalog(registry, max_chars=1)

    def test_model_skill_route_rejects_archived_names_and_skips_an_empty_active_catalog(self):
        from src.runtime.runner import TaskRunner, BudgetExceeded
        body = "## Procedure\n1. Read.\n2. Compare.\n3. Explain."
        state = ArtifactState("route-test", "Answer from evidence.", Registry("test", [
            Skill("old", "G-main", "assay", body=body, lifecycle="retired"),
            Skill("current", "G-main", "assay", body=body)]))
        runner = TaskRunner.__new__(TaskRunner)
        runner.state = state
        with patch.object(runner, "_meta_call", return_value="SKILL: old"):
            with self.assertRaises(BudgetExceeded):
                runner._model_skill({"question": "assay"}, "parent", [])
        state.registry.skills[1].lifecycle = "retired"
        with patch.object(runner, "_meta_call") as model:
            self.assertIsNone(runner._model_skill({"question": "assay"}, "parent", []))
            model.assert_not_called()

    def test_bootstrap_archive_refuses_path_escape_and_symbolic_links(self):
        from scripts.bootstrap_project import unpack_archive
        from io import BytesIO
        import zipfile
        for name in ("../escape", "/absolute", "source/../../escape", "C:/escape"):
            content = BytesIO()
            with zipfile.ZipFile(content, "w") as archive:
                archive.writestr(name, "content")
            with tempfile.TemporaryDirectory() as directory, self.assertRaises(ValueError):
                unpack_archive(content.getvalue(), Path(directory))
