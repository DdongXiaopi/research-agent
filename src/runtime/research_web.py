"""Local single-user HTTP interface for the research workflow."""
from __future__ import annotations

import json
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from uuid import uuid4

from ..evolution.session_feedback import record_feedback
from . import research_workflow

ASSETS = Path(__file__).resolve().parents[1] / "web"
PROVIDERS = {"crossref", "openalex", "semantic_scholar", "europe_pmc", "pubmed"}


class ResearchServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, port, cfg):
        super().__init__(("127.0.0.1", port), Handler)
        self.cfg = cfg
        self.token = secrets.token_urlsafe(32)
        self.lock = threading.RLock()
        self.jobs = {}
        self.busy = False

    @property
    def origin(self):
        return f"http://127.0.0.1:{self.server_address[1]}"

    def submit(self, body):
        if not isinstance(body, dict) or set(body) - {"query", "source_text", "providers"}:
            raise ValueError("invalid task fields")
        query, source_text, providers = body.get("query"), body.get("source_text", ""), body.get("providers", [])
        if not isinstance(query, str) or not query.strip() or len(query) > 2000:
            raise ValueError("question must contain 1–2000 characters")
        if not isinstance(source_text, str) or len(source_text) > 60000:
            raise ValueError("source text must contain at most 60000 characters")
        if (not isinstance(providers, list) or any(not isinstance(p, str) for p in providers)
                or len(set(providers)) != len(providers) or set(providers) - PROVIDERS):
            raise ValueError("invalid literature sources")
        if bool(source_text.strip()) == bool(providers):
            raise ValueError("choose either supplied source text or literature sources")
        with self.lock:
            if self.busy:
                raise ValueError("a task is already running")
            if len(self.jobs) >= 100:
                raise ValueError("task history is full; restart the local server")
            self.busy = True
            job_id = uuid4().hex
            self.jobs[job_id] = {"job_id": job_id, "status": "running", "stage": "parse", "stages_completed": 0}
        value = {"task_id": "web-" + job_id, "task_type": "open_research", "query": query.strip(),
                 "project_id": "web", "session_id": "session-" + job_id,
                 "evidence": ([{"source_id": "user-source", "text": source_text.strip(),
                                "locator": "user-supplied material"}] if source_text.strip() else [])}

        def update(event):
            with self.lock:
                self.jobs[job_id].update(stage=event["stage"], stages_completed=event["stages_completed"],
                                         workflow_dir=event["workflow_dir"])

        def run():
            try:
                plan = research_workflow.build_plan(value, self.cfg, providers=providers, limit=3,
                                                     max_requests=6, retries=0, timeout=15)
                result = research_workflow.execute(value, self.cfg, plan["plan_sha256"], providers=providers,
                    limit=3, max_requests=6, retries=0, timeout=15, on_stage=update)
                with self.lock:
                    self.jobs[job_id].update(status="completed", result=result)
            except Exception as exc:
                with self.lock:
                    self.jobs[job_id].update(status="failed", error_type=type(exc).__name__)
            finally:
                with self.lock:
                    self.busy = False
        threading.Thread(target=run, daemon=True).start()
        return {"job_id": job_id}

    def feedback(self, body):
        if not isinstance(body, dict) or set(body) - {"job_id", "category", "correction", "reviewer", "consent", "source_id", "quote"}:
            raise ValueError("invalid feedback fields")
        with self.lock:
            job = self.jobs.get(body.get("job_id"))
            if not job or job.get("status") != "completed":
                raise ValueError("feedback requires a completed task")
            run_dir = job["result"]["run_dir"]
        if type(body.get("consent")) is not bool:
            raise ValueError("feedback consent must be explicit")
        quote, source_id = body.get("quote", ""), body.get("source_id", "")
        if not isinstance(quote, str) or not isinstance(source_id, str) or bool(quote.strip()) != bool(source_id.strip()):
            raise ValueError("source and quotation must be supplied together")
        quotes = [{"source_id": source_id.strip(), "quote": quote.strip()}] if quote.strip() else []
        path = record_feedback(Path(run_dir), category=body.get("category"), correction=body.get("correction"),
            reviewer=body.get("reviewer"), use_for_evolution=body["consent"], evidence_quotes=quotes)
        return {"status": "saved", "feedback_id": path.stem}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _valid_host(self):
        return self.headers.get("Host") == f"127.0.0.1:{self.server.server_address[1]}"

    def _respond(self, status, body, content_type="application/json; charset=utf-8"):
        payload = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; base-uri 'none'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if not self._valid_host():
            return self._respond(403, {"error": "invalid host"})
        if self.path == "/api/session":
            return self._respond(200, {"token": self.server.token})
        if self.path.startswith("/api/jobs/"):
            job_id = self.path.removeprefix("/api/jobs/")
            with self.server.lock:
                job = self.server.jobs.get(job_id)
                data = json.loads(json.dumps(job, ensure_ascii=False)) if job else None
            return self._respond(200 if data else 404, data or {"error": "task not found"})
        assets = {"/": ("index.html", "text/html; charset=utf-8"),
                  "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                  "/style.css": ("style.css", "text/css; charset=utf-8")}
        if self.path not in assets:
            return self._respond(404, {"error": "not found"})
        name, mime = assets[self.path]
        return self._respond(200, (ASSETS / name).read_bytes(), mime)

    def do_POST(self):
        if (not self._valid_host() or self.headers.get("Origin") != self.server.origin
                or not secrets.compare_digest(self.headers.get("X-Research-Token", ""), self.server.token)):
            # On Windows, closing a connection with unread request bytes may
            # reset the TCP stream before the client receives the 403. Drain
            # only a bounded body and close; never wait indefinitely on a peer.
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if 0 < size <= 260000:
                    self.connection.settimeout(5)
                    self.rfile.read(size)
            except (ValueError, OSError):
                pass
            self.close_connection = True
            return self._respond(403, {"error": "invalid local session"})
        try:
            size = int(self.headers.get("Content-Length", "-1"))
            if not 0 < size <= 260000:
                raise ValueError("request size is invalid")
            if self.headers.get_content_type() != "application/json":
                raise ValueError("expected JSON")
            value = json.loads(self.rfile.read(size))
            if self.path == "/api/tasks":
                result = self.server.submit(value)
            elif self.path == "/api/feedback":
                result = self.server.feedback(value)
            else:
                return self._respond(404, {"error": "not found"})
            self._respond(200, result)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self._respond(400, {"error": str(exc)[:300]})

