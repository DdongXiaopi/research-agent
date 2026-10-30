"""Bounded, license-declared OA full text and supplied-document retrieval."""
from __future__ import annotations

import io
import json
import re
import time
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse, urljoin
from uuid import uuid4
import xml.etree.ElementTree as ET

import requests

from ..util import artifacts_path, now_iso, sha256_obj, sha256_text, sha256_file, write_json
from .literature import validate_snapshot

OA_HOSTS = frozenset({"pmc.ncbi.nlm.nih.gov", "www.ncbi.nlm.nih.gov", "europepmc.org",
    "www.ebi.ac.uk", "arxiv.org", "export.arxiv.org", "journals.plos.org", "elifesciences.org"})
OA_LICENSES = frozenset({"cc-by", "cc-by-4.0", "cc-by-3.0", "cc0", "cc0-1.0",
    "cc-by-sa", "cc-by-sa-4.0", "public-domain"})
PLOS_CDN = "corpus-prod.f505aadad1e46610c8f8563037eb3c4b.r2.cloudflarestorage.com"


def _url(value):
    parsed = urlparse(value)
    publisher_cdn = parsed.hostname == PLOS_CDN and parsed.path.startswith("/10.1371/")
    if (parsed.scheme != "https" or (parsed.hostname not in OA_HOSTS and not publisher_cdn) or parsed.username
            or parsed.password or parsed.port not in (None, 443) or parsed.fragment):
        raise ValueError("full text requires HTTPS on an approved public OA source host")
    return value


def build_plan(snapshot, source_id, *, max_bytes=8_000_000, timeout=20, max_chars=500000):
    from ..eval.run_manifest import source_tree_fingerprint
    from importlib.metadata import version, PackageNotFoundError
    try:
        parser = version("pypdf")
    except PackageNotFoundError:
        parser = None
    validate_snapshot(snapshot)
    matches = [x for x in snapshot["records"] if x["source_id"] == source_id]
    if len(matches) != 1:
        raise ValueError("full text source must identify one frozen metadata record")
    record = matches[0]
    if record.get("oa_license", "").lower() not in OA_LICENSES:
        raise ValueError("full text requires an explicit supported open license")
    url = _url(record.get("oa_url", ""))
    if (type(max_bytes) is not int or not 1000 <= max_bytes <= 8_000_000
            or type(timeout) is not int or not 1 <= timeout <= 30
            or type(max_chars) is not int or not 100 <= max_chars <= 500000):
        raise ValueError("invalid full text byte/deadline/character budget")
    plan = {"schema": "oa-fulltext-plan-v1", "metadata_snapshot_sha256": snapshot["snapshot_sha256"],
            "source_tree_sha256": source_tree_fingerprint()["sha256"], "pdf_parser_version": parser,
            "source_id": source_id, "record_sha256": sha256_obj(record), "url": url,
            "license": record["oa_license"], "max_bytes": max_bytes, "timeout": timeout,
            "max_chars": max_chars, "max_requests": 4, "model_calls": 0}
    return {**plan, "plan_sha256": sha256_obj(plan)}


class _HTMLText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hidden = 0
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.hidden += 1

    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self.hidden = max(0, self.hidden - 1)

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def _extract(payload, content_type, root, plan):
    if payload.startswith(b"%PDF"):
        target = root / "source.pdf"
        target.write_bytes(payload)
        worker = Path(__file__).with_name("pdf_text_worker.py")
        run = subprocess.run([sys.executable, str(worker), str(target), str(plan["max_chars"])],
                             capture_output=True, text=True, encoding="utf-8", timeout=plan["timeout"])
        if run.returncode:
            raise ValueError("PDF extraction failed; install requirements-paperqa.txt or inspect receipt")
        return run.stdout, "pdf"
    text = payload.decode("utf-8-sig", errors="strict")
    if "xml" in content_type or text.lstrip().startswith("<?xml"):
        if "<!ENTITY" in text.upper():
            raise ValueError("XML entity declarations are not accepted")
        # ElementTree does not fetch external DTDs. Reject explicit entity
        # declarations while accepting the normal publisher JATS DOCTYPE.
        document = ET.fromstring(text)
        body = next((node for node in document.iter() if node.tag.rsplit("}", 1)[-1] == "body"), document)
        return " ".join(body.itertext()), "xml"
    if "html" in content_type or text.lstrip().lower().startswith(("<!doctype html", "<html")):
        parser = _HTMLText()
        parser.feed(text)
        return " ".join(parser.parts), "html"
    if content_type.startswith("text/plain"):
        return text, "text"
    raise ValueError("unsupported full text format; expected PDF, UTF-8 XML/HTML/plain text")


def execute(snapshot, plan, expected_plan_sha256, *, session=None):
    if (expected_plan_sha256 != plan.get("plan_sha256") or build_plan(snapshot, plan["source_id"],
            max_bytes=plan["max_bytes"], timeout=plan["timeout"], max_chars=plan["max_chars"]) != plan):
        raise ValueError("full text differs from caller-reviewed metadata/plan")
    root = artifacts_path("research_tools", "fulltext-" + uuid4().hex)
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "plan.json", plan)
    write_json(root / "metadata.json", snapshot)
    own_session = session is None
    session = session or requests.Session()
    events, url = [], plan["url"]
    started = time.monotonic()
    try:
        for index in range(plan["max_requests"]):
            _url(url)
            parsed_url = urlparse(url)
            safe_url = parsed_url._replace(query="").geturl() if parsed_url.hostname == PLOS_CDN else url
            event = {"attempt": index + 1, "url": safe_url, "url_sha256": sha256_text(url), "status": "dispatching"}
            events.append(event)
            write_json(root / "requests.json", events)
            remaining = plan["timeout"] - (time.monotonic() - started)
            if remaining <= 0:
                raise ValueError("full text deadline exhausted")
            response = session.get(url, timeout=remaining, stream=True, allow_redirects=False,
                                   headers={"User-Agent": "rsi-oa-document-reader/1.0"})
            try:
                event["http_status"] = response.status_code
                if response.status_code in {301, 302, 303, 307, 308}:
                    url = _url(urljoin(url, response.headers.get("Location", "")))
                    event["status"] = "redirect"
                    continue
                if response.status_code != 200:
                    raise ValueError(f"full text HTTP {response.status_code}")
                payload = bytearray()
                for block in response.iter_content(65536):
                    if len(payload) + len(block) > plan["max_bytes"] or time.monotonic() - started > plan["timeout"]:
                        raise ValueError("full text response exceeded byte/deadline budget")
                    payload.extend(block)
                remaining = plan["timeout"] - (time.monotonic() - started)
                if remaining <= 0:
                    raise ValueError("full text deadline exhausted")
                text, format = _extract(bytes(payload), response.headers.get("Content-Type", ""), root,
                                        {**plan, "timeout": remaining})
                text = re.sub(r"\s+", " ", text).strip()
                if not text or len(text) > plan["max_chars"]:
                    raise ValueError("full text is empty or exceeds the character budget")
                (root / "source.bin").write_bytes(payload)
                event["status"] = "ok"
                write_json(root / "requests.json", events)
                record = next(x for x in snapshot["records"] if x["source_id"] == plan["source_id"])
                doc = {"source_id": record["source_id"], "text": text, "url": plan["url"],
                       "title": record["title"], "license": plan["license"], "doi": record.get("doi", ""),
                       "retrieved_at": now_iso(), "locator": "OA full text: " + format,
                       "content_sha256": sha256_text(text), "download_sha256": sha256_file(root / "source.bin")}
                result = {"schema": "fulltext-snapshot-v1", "plan": plan, "documents": [doc],
                          "events": events, "physical_requests": len(events), "model_calls": 0,
                          "run_dir": str(root), "status": "completed", "full_text_downloaded": True}
                result["snapshot_sha256"] = sha256_obj(result)
                write_json(root / "snapshot.json", result)
                write_json(root / "status.json", {"status": "completed", "snapshot_sha256": result["snapshot_sha256"]})
                return result
            finally:
                response.close()
        raise ValueError("full text redirect budget exhausted")
    except BaseException as exc:
        write_json(root / "requests.json", events)
        write_json(root / "status.json", {"status": "failed", "error_type": type(exc).__name__,
                                         "physical_requests": len(events)})
        raise
    finally:
        if own_session:
            session.close()


def evidence(snapshot):
    if (snapshot.get("schema") != "fulltext-snapshot-v1" or snapshot.get("status") != "completed"
            or snapshot.get("snapshot_sha256") != sha256_obj({k: v for k, v in snapshot.items() if k != "snapshot_sha256"})):
        raise ValueError("full text snapshot identity changed")
    output = []
    for doc in snapshot["documents"]:
        if (doc.get("content_sha256") != sha256_text(doc["text"])
                or doc.get("license", "").lower() not in OA_LICENSES):
            raise ValueError("full text content/license changed")
        output.append({key: doc[key] for key in ("source_id", "text", "url", "locator", "retrieved_at")})
    if not output:
        raise ValueError("full text snapshot contains no documents")
    return output


def retrieval_snapshot(snapshot, query, *, backend="local", top_k=5):
    if not isinstance(query, str) or not query.strip() or len(query) > 8000:
        raise ValueError("document retrieval requires a bounded query")
    sources = evidence(snapshot)
    if backend == "paperqa":
        from .gateway import build_paperqa_gateway
        from importlib.metadata import version
        gateway = build_paperqa_gateway(top_k, sources)
        docs = gateway.search({"question": query, "options": []})
        selected = [{"source_id": doc["doc_id"], "text": doc["text"], "url": doc["url"],
                     "locator": doc["locator"], "retrieved_at": doc["retrieved_at"]} for doc in docs]
        adapter = {"name": "paperqa-native-records", "version": version("paper-qa")}
    elif backend == "local":
        if type(top_k) is not int or not 1 <= top_k <= 20:
            raise ValueError("document retrieval top_k must be 1 through 20")
        from .chunking import chunk_text
        from .corpus import tokens
        chunks = []
        for source in sources:
            for index, text in enumerate(chunk_text(source["text"])):
                chunks.append({**source, "source_id": "doc-" + sha256_obj([source["source_id"], index, text])[:24],
                               "text": text, "locator": source["locator"] + f"; chunk {index + 1}"})
        selected = sorted(chunks, key=lambda x: (-len(tokens(x["text"]) & tokens(query)), x["source_id"]))[:top_k]
        adapter = {"name": "local-lexical", "version": "1"}
    else:
        raise ValueError("document backend must be local or paperqa")
    result = {"schema": "document-evidence-snapshot-v1", "source_snapshot": snapshot,
              "source_snapshot_sha256": snapshot["snapshot_sha256"], "query": query,
              "adapter": adapter, "top_k": top_k, "evidence": selected,
              "model_calls": 0, "physical_requests": 0}
    result["snapshot_sha256"] = sha256_obj(result)
    return result


def selected_evidence(snapshot):
    if (snapshot.get("snapshot_sha256") != sha256_obj({k: v for k, v in snapshot.items() if k != "snapshot_sha256"})
            or snapshot.get("source_snapshot_sha256") != snapshot.get("source_snapshot", {}).get("snapshot_sha256")):
        raise ValueError("document retrieval snapshot identity changed")
    evidence(snapshot["source_snapshot"])
    # Recreate the exact zero-call retrieval so edited chunks cannot be rehashed as original evidence.
    backend = "paperqa" if snapshot["adapter"]["name"] == "paperqa-native-records" else "local"
    expected = retrieval_snapshot(snapshot["source_snapshot"], snapshot["query"],
                                  backend=backend, top_k=snapshot["top_k"])
    if expected != snapshot:
        raise ValueError("document retrieval output differs from frozen source/query/adapter")
    return snapshot["evidence"]
