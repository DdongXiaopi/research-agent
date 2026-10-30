"""The browser API runs the real workflow and persists consented feedback."""
import threading
import time

import requests

from src.config import load_config
from src.governance import release
from src.runtime.research_web import ResearchServer
from src.util import artifacts_path, read_json


def test_local_ui_task_feedback_and_request_origin_guards():
    release.init_baseline()
    server = ResearchServer(0, load_config("configs/experiment.yaml"))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = server.origin
        page = requests.get(url + "/", timeout=5)
        assert page.status_code == 200 and "研究问题" in page.text
        token = requests.get(url + "/api/session", timeout=5).json()["token"]
        headers = {"Origin": url, "X-Research-Token": token}
        body = {"query": "What was used?", "source_text": "Two controls were used.", "providers": []}
        assert requests.post(url + "/api/tasks", json=body, timeout=5).status_code == 403
        assert requests.post(url + "/api/tasks", json=body, headers={**headers, "Origin": "http://evil.test"}, timeout=5).status_code == 403
        assert requests.get(url + "/", headers={"Host": "evil.test"}, timeout=5).status_code == 403
        response = requests.post(url + "/api/tasks", json=body, headers=headers, timeout=5)
        assert response.status_code == 200
        job_id = response.json()["job_id"]
        deadline = time.monotonic() + 40
        while time.monotonic() < deadline:
            job = requests.get(url + "/api/jobs/" + job_id, timeout=5).json()
            if job["status"] != "running":
                break
            time.sleep(0.1)
        assert job["status"] == "completed", job
        assert len(job["result"]["stages"]) == 6
        assert job["result"]["citation_verification"]["status"] == "identity_verified"
        feedback = {"job_id": job_id, "category": "evidence_fact", "correction": "There were two controls.",
                    "reviewer": "researcher", "consent": True, "source_id": "user-source", "quote": "Two controls were used."}
        saved = requests.post(url + "/api/feedback", json=feedback, headers=headers, timeout=5)
        assert saved.status_code == 200
        record = read_json(artifacts_path("research_feedback", "web", "evidence_fact", saved.json()["feedback_id"] + ".json"))
        assert record["use_for_evolution"] is True and record["evidence_quotes"][0]["quote"] == feedback["quote"]
        feedback["quote"] = "fabricated"
        assert requests.post(url + "/api/feedback", json=feedback, headers=headers, timeout=5).status_code == 400
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
