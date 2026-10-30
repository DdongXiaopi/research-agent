from __future__ import annotations
import copy
import json
from types import SimpleNamespace
from unittest.mock import patch
import pytest

from src.config import load_config, config_fingerprint
from src.data import public_tasks, split, freeze
from src.eval import controlled_experiment as experiment
from src.governance import release
from src.retrieval import documents
from src.util import sha256_obj, write_json, write_jsonl


def rows():
    return [{"task_id": f"public-{i}", "question": f"What is the reported marker in case {i}?",
             "options": ["alpha marker", "beta marker"], "correct_index": i % 2,
             "evidence": f"Synthetic case {i} reports {'alpha' if i % 2 == 0 else 'beta'} marker.",
             "sources": [f"synthetic://article/{i}"]} for i in range(100)]


def test_public_protocol_has_no_historical_manifest_or_optimizer_dependency():
    cfg = load_config("configs/experiment.public-task-input.yaml")
    with patch("src.config._locked_upstream_source_trees", side_effect=AssertionError("unused vendor")), \
         patch("src.config.TASK_INPUT_PROTOCOL_SOURCES", ("data/missing-history.json",)):
        assert len(config_fingerprint(cfg)) == 64
        legacy = copy.deepcopy(cfg)
        legacy["runtime"].pop("protocol_epoch")
        with pytest.raises(FileNotFoundError):
            config_fingerprint(legacy)


def test_public_source_components_are_kept_in_one_partition():
    tasks = rows()
    tasks[1]["sources"] = tasks[0]["sources"]
    tasks[2]["evidence"] = tasks[1]["evidence"]
    _, visible, sealed = public_tasks.prepare("public_test", tasks, source="synthetic", license="MIT")
    assignments = {x["task_id"]: x["split"] for x in visible + sealed}
    assert assignments["public-0"] == assignments["public-1"] == assignments["public-2"]
    for row in tasks:
        row["sources"] = ["synthetic://same-article"]
    with pytest.raises(ValueError, match="independent source groups"):
        public_tasks.prepare("public_test", tasks, source="synthetic", license="MIT")


def test_public_source_identity_normalizes_whitespace_before_grouping():
    tasks = rows()
    tasks[1]["sources"] = ["  " + tasks[0]["sources"][0] + "  "]
    tasks[2]["evidence"] = " \n ".join(tasks[0]["evidence"].split())
    tasks[3]["group_id"], tasks[4]["group_id"] = " shared-group ", "shared-group"
    _, visible, sealed = public_tasks.prepare("public_test", tasks, source="synthetic", license="MIT")
    prepared = {row["task_id"]: row for row in visible + sealed}
    assert len({prepared[f"public-{i}"]["group_id"] for i in (0, 1, 2)}) == 1
    assert prepared["public-3"]["group_id"] == prepared["public-4"]["group_id"]


def test_public_tasks_reject_ambiguous_options_and_invalid_domain():
    tasks = rows()
    tasks[0]["options"] = ["alpha", " alpha "]
    with pytest.raises(ValueError, match="unique options"):
        public_tasks.prepare("public_test", tasks, source="synthetic", license="MIT")
    tasks = rows()
    tasks[0]["domain"] = {"unexpected": "object"}
    with pytest.raises(ValueError, match="domain"):
        public_tasks.prepare("public_test", tasks, source="synthetic", license="MIT")


def test_crystallization_does_not_ignore_a_failed_negative_dataset():
    from src.evolution import crystallize
    cfg = {"evaluation": {"datasets": ["target", "negative_a", "negative_b"]}}
    def load(dataset, **kwargs):
        if dataset == "negative_b":
            raise RuntimeError("unavailable")
        return [{"task_id": "n", "dataset": dataset}]
    with patch("src.data.split.load_tasks", side_effect=load):
        with pytest.raises(crystallize.CrystallizationError, match="negative_b"):
            crystallize.cross_domain_negative_tasks(cfg, ["target"])


@pytest.mark.parametrize("module_name,loader_name", [
    ("src.evolution.counterfactual", "task_index"),
    ("src.evolution.reflection", "_task_index"),
    ("src.governance.policy", "_visible_demo_tasks"),
])
def test_candidate_task_loaders_recheck_data_on_each_use(module_name, loader_name):
    import importlib
    module = importlib.import_module(module_name)
    loader = getattr(module, loader_name)
    target = "src.data.split.load_tasks" if loader_name == "_visible_demo_tasks" else module_name + ".load_tasks"
    item = {"task_id": "x", "dataset": "public_test", "split": "dev_optimize"}
    argument = "public_test" if loader_name == "_visible_demo_tasks" else ["public_test"]
    with patch(target, side_effect=[[item], RuntimeError("data integrity changed")]):
        assert "x" in loader(argument)
        with pytest.raises(RuntimeError, match="integrity changed"):
            loader(argument)


def test_corpus_cannot_read_sealed_pool_without_unlock_even_after_cache(tmp_path, monkeypatch):
    from src.data.split import FinalHoldoutSealed
    from src.retrieval import corpus
    monkeypatch.setattr(corpus, "path", lambda *parts: tmp_path.joinpath(*parts))
    monkeypatch.delenv("RSI_ALLOW_FINAL", raising=False)
    corpus._POOL_CACHE.clear()
    write_jsonl(tmp_path / "data/normalized/public_test.jsonl", [
        {"task_id": "dev", "split": "dev_optimize", "evidence": "dev evidence " * 5},
        {"task_id": "misplaced-final", "split": "final_holdout", "evidence": "sealed evidence " * 5}])
    write_jsonl(tmp_path / "data/sealed/public_test.final_holdout.jsonl", [
        {"task_id": "sealed", "split": "final_holdout", "evidence": "sealed evidence " * 5}])
    assert [row["task_id"] for row in corpus.load_pool("public_test", exclude_splits=())] == ["dev"]
    with pytest.raises(FinalHoldoutSealed):
        corpus.load_pool("public_test", allow_all=True)
    monkeypatch.setenv("RSI_ALLOW_FINAL", "1")
    assert "sealed" in {row["task_id"] for row in corpus.load_pool("public_test", allow_all=True)}
    monkeypatch.delenv("RSI_ALLOW_FINAL")
    with pytest.raises(FinalHoldoutSealed):
        corpus.load_pool("public_test", allow_all=True)


def test_corpus_cache_tracks_bytes_and_cannot_be_modified_by_callers(tmp_path, monkeypatch):
    import os
    from src.retrieval import corpus
    monkeypatch.setattr(corpus, "path", lambda *parts: tmp_path.joinpath(*parts))
    corpus._POOL_CACHE.clear()
    source = tmp_path / "data/normalized/public_test.jsonl"
    row = {"task_id": "dev", "split": "dev_optimize", "evidence": "evidence alpha " * 4}
    write_jsonl(source, [row])
    stat = source.stat()
    first = corpus.load_pool("public_test")
    first[0]["text"] = "modified by caller"
    assert corpus.load_pool("public_test")[0]["text"] == row["evidence"].strip()
    write_jsonl(source, [{**row, "evidence": "evidence bravo " * 4}])
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert corpus.load_pool("public_test")[0]["text"] == ("evidence bravo " * 4).strip()


@pytest.fixture
def public_cohort(tmp_path, monkeypatch):
    for module in (split, freeze):
        monkeypatch.setattr(module, "MANIFEST_DIR", tmp_path / "manifests")
        monkeypatch.setattr(module, "NORMALIZED_DIR", tmp_path / "normalized")
        monkeypatch.setattr(module, "SEALED_DIR", tmp_path / "sealed")
    monkeypatch.setattr(freeze, "FREEZE_FILE", tmp_path / "manifests/freeze.json")
    dataset = "public_" + sha256_obj(str(tmp_path))[:12]
    manifest, visible, sealed = public_tasks.prepare(dataset, rows(), source="synthetic", license="MIT")
    write_json(tmp_path / f"manifests/{dataset}.json", manifest)
    write_jsonl(tmp_path / f"normalized/{dataset}.jsonl", visible)
    write_jsonl(tmp_path / f"sealed/{dataset}.final_holdout.jsonl", sealed)
    freeze.freeze([dataset])
    cfg = load_config("configs/experiment.public-task-input.yaml")
    cfg["evaluation"].update(datasets=[dataset], manifests=[dataset + ".json"], primary_final_datasets=[dataset])
    return cfg, dataset


@pytest.mark.parametrize("kind", ["multiseed", "ablate"])
def test_visible_experiments_are_plan_bound_and_one_shot(public_cohort, kind):
    cfg, dataset = public_cohort
    rid = release.current_release_id()
    spec = {"base": rid, "compare": None} if kind == "multiseed" else {
        "base": rid, "conditions": {"A0": rid, "A1": rid}}
    plan = experiment.build_plan(cfg, kind=kind, spec=spec, datasets=[dataset],
                                 splits=["dev_optimize"], seeds=[11, 12, 13], limit=2)
    with pytest.raises(ValueError, match="caller-reviewed"):
        experiment.execute(cfg, plan, "0" * 64)
    result = experiment.execute(cfg, plan, plan["plan_sha256"])
    assert result["status"] == "completed"
    assert result["request_accounting"]["physical_http_attempts"] == 0
    with pytest.raises(FileExistsError):
        experiment.execute(cfg, plan, plan["plan_sha256"])
    with pytest.raises(ValueError, match="visible splits"):
        experiment.build_plan(cfg, kind=kind, spec=spec, datasets=[dataset],
                              splits=["final_holdout"], seeds=[11, 12, 13], limit=2)


def metadata(url="https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.fixture&type=manuscript", license="cc-by"):
    snapshot = {"schema": "literature-snapshot-v1", "records": [{"source_id": "lit-fixture", "title": "Synthetic paper",
                "oa_url": url, "oa_license": license, "doi": "10.1371/journal.pone.fixture"}]}
    snapshot["snapshot_sha256"] = sha256_obj(snapshot)
    return snapshot


def test_licensed_fulltext_reaches_retrieval_and_rejects_tampering():
    payload = b'<?xml version="1.0"?><article><body><p>The assay lacked a control group.</p></body></article>'
    response = SimpleNamespace(status_code=200, headers={"Content-Type": "application/xml"},
                               iter_content=lambda **kw: iter([payload]), close=lambda: None)
    # Accept requests' positional chunk-size argument too.
    response.iter_content = lambda *args, **kw: iter([payload])
    session = SimpleNamespace(get=lambda *args, **kwargs: response)
    snapshot = metadata()
    plan = documents.build_plan(snapshot, "lit-fixture")
    fetched = documents.execute(snapshot, plan, plan["plan_sha256"], session=session)
    selected = documents.retrieval_snapshot(fetched, "control group")
    assert "lacked a control group" in documents.selected_evidence(selected)[0]["text"]
    selected["evidence"][0]["text"] = "fabricated source"
    selected["snapshot_sha256"] = sha256_obj({k: v for k, v in selected.items() if k != "snapshot_sha256"})
    with pytest.raises(ValueError, match="differs from frozen"):
        documents.selected_evidence(selected)
    with pytest.raises(ValueError, match="open license"):
        documents.build_plan(metadata(license="unknown"), "lit-fixture")
    with pytest.raises(ValueError, match="approved public"):
        documents.build_plan(metadata(url="https://127.0.0.1/private"), "lit-fixture")


def test_fulltext_private_redirect_is_rejected_before_second_request():
    response = SimpleNamespace(status_code=302, headers={"Location": "https://127.0.0.1/secret"}, close=lambda: None)
    calls = []
    session = SimpleNamespace(get=lambda *args, **kwargs: calls.append(args) or response)
    snapshot = metadata()
    plan = documents.build_plan(snapshot, "lit-fixture")
    with pytest.raises(ValueError, match="approved public"):
        documents.execute(snapshot, plan, plan["plan_sha256"], session=session)
    assert len(calls) == 1


def test_native_paperqa_records_reach_existing_retrieval_interface():
    pytest.importorskip("paperqa")
    from src.retrieval.gateway import build_paperqa_gateway
    gateway = build_paperqa_gateway(documents=[{"source_id": "paper", "text": "The assay lacks controls.",
                                               "url": "https://example.org/paper"}])
    result = gateway.search({"question": "assay controls", "options": []})
    assert result[0]["source_kind"] == "paperqa_document"
    assert gateway.records[0].doc.content_hash


def test_measured_proposal_keeps_raw_and_normalizes_only_transport_newlines(tmp_path):
    from src.evolution.crystallize import propose_skill_from_failures
    from src.runtime.skills import parse_procedure
    skill = {"name": "transport-procedure-check", "description": "Check assay controls",
             "triggers": ["assay", "control"], "body": r"Procedure\n1. Read evidence.\n2. Check controls.\n3. Cite sources."}
    provider = SimpleNamespace(chat=lambda *args, **kwargs: json.dumps(skill))
    cfg = {"models": {"reflection": {"provider": "http", "model_id": "fixture"}}}
    result = propose_skill_from_failures(release.load_state(), [{"split": "dev_optimize"}], cfg,
        provider=provider, cases_text="synthetic fixture", audit_dir=tmp_path / "audit")
    assert len(parse_procedure(result.body)) == 3
    raw = json.loads((tmp_path / "audit/raw-response.json").read_text())
    assert json.loads(raw["content"])["body"] == skill["body"]
    assert (tmp_path / "audit/normalization.json").exists()


def test_publisher_cdn_jats_redirect_preserves_canonical_source_and_redacts_signature():
    payload = b'<?xml version="1.0"?><!DOCTYPE article SYSTEM "https://example.org/non-fetched.dtd"><article><body><p>Published open evidence.</p></body></article>'
    first = SimpleNamespace(status_code=302, headers={"Location": "https://" + documents.PLOS_CDN +
        "/10.1371/journal.pone.fixture/1/paper.xml?X-Amz-Signature=synthetic-test"}, close=lambda: None)
    second = SimpleNamespace(status_code=200, headers={"Content-Type": "application/xml"},
        iter_content=lambda *args, **kwargs: iter([payload]), close=lambda: None)
    responses = iter([first, second])
    session = SimpleNamespace(get=lambda *args, **kwargs: next(responses))
    snapshot = metadata()
    plan = documents.build_plan(snapshot, "lit-fixture")
    result = documents.execute(snapshot, plan, plan["plan_sha256"], session=session)
    assert result["physical_requests"] == 2
    assert result["documents"][0]["url"] == plan["url"]
    assert all("X-Amz" not in event["url"] for event in result["events"])


def test_live_acceptance_price_parser_blocks_nonzero_models():
    from src.runtime.free_model import parse_zero_price
    def page(price):
        stream = '\n'.join(['1:' + json.dumps({"modelName": "fixture-model", "status": "normal", "pricing": "$2"}),
            '2:["$3","$4"]', '3:' + json.dumps({"specification": "prompt", "price": "0"}),
            '4:' + json.dumps({"specification": "completion", "price": price})])
        return '<script>self.__next_f.push([1,' + json.dumps(stream) + '])</script>'
    assert parse_zero_price(page("0"), "fixture-model")["completion_price"] == "0"
    with pytest.raises(ValueError, match="not currently zero-priced"):
        parse_zero_price(page("0.1"), "fixture-model")


def test_proposal_json_literal_newlines_are_audited_and_duplicate_keys_rejected(tmp_path):
    from src.evolution.crystallize import propose_skill_from_failures
    from src.runtime.skills import parse_procedure
    skill = {"name": "literal-newline-procedure", "description": "Check source controls",
             "triggers": ["source", "control"], "body": "## Procedure\n1. Read.\n2. Check.\n3. Cite."}
    raw = json.dumps(skill).replace("\\n", "\n")
    cfg = {"models": {"reflection": {"provider": "http", "model_id": "fixture"}}}
    provider = SimpleNamespace(chat=lambda *args, **kwargs: raw)
    result = propose_skill_from_failures(release.load_state(), [{"split": "dev_optimize"}], cfg,
        provider=provider, cases_text="synthetic", audit_dir=tmp_path / "literal")
    assert len(parse_procedure(result.body)) == 3
    assert (tmp_path / "literal/json-transport-normalization.json").exists()
    duplicate = json.dumps(skill)[:-1] + ', "name":"different-name"}'
    provider.chat = lambda *args, **kwargs: duplicate
    with pytest.raises(ValueError, match="unambiguous JSON"):
        propose_skill_from_failures(release.load_state(), [{"split": "dev_optimize"}], cfg,
            provider=provider, cases_text="synthetic", audit_dir=tmp_path / "duplicate")


def test_optional_pdf_parser_child_returns_source_bound_text():
    pypdf = pytest.importorskip("pypdf")
    import io
    from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject
    writer = pypdf.PdfWriter()
    page = writer.add_blank_page(width=300, height=300)
    font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
        NameObject("/Subtype"): NameObject("/Type1"), NameObject("/BaseFont"): NameObject("/Helvetica")})
    page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"):
        DictionaryObject({NameObject("/F1"): writer._add_object(font)})})
    stream = DecodedStreamObject()
    stream.set_data(b"BT /F1 12 Tf 20 100 Td (Synthetic PDF control evidence.) Tj ET")
    page[NameObject("/Contents")] = writer._add_object(stream)
    buffer = io.BytesIO()
    writer.write(buffer)
    response = SimpleNamespace(status_code=200, headers={"Content-Type": "application/pdf"},
        iter_content=lambda *args, **kwargs: iter([buffer.getvalue()]), close=lambda: None)
    snapshot = metadata()
    plan = documents.build_plan(snapshot, "lit-fixture")
    result = documents.execute(snapshot, plan, plan["plan_sha256"],
        session=SimpleNamespace(get=lambda *args, **kwargs: response))
    assert "Synthetic PDF control evidence" in documents.evidence(result)[0]["text"]
