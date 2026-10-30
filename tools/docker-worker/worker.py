"""Trusted stdin/stdout renderer. No user code, host mount or network needed."""
import base64
import hashlib
import json
import os
from pathlib import Path
import resource
import socket
import subprocess
import sys

from worker_contract import SCHEMA, MAX_REQUEST_BYTES, MAX_ARTIFACT_BYTES, MAX_TOTAL_BYTES, validate_request


def artifact(file, media):
    if not 0 < file.stat().st_size <= MAX_ARTIFACT_BYTES:
        raise ValueError("rendered artifact exceeds output budget")
    data = file.read_bytes()
    if not data or len(data) > MAX_ARTIFACT_BYTES:
        raise ValueError("rendered artifact exceeds output budget")
    return {"name": file.name, "media_type": media, "size": len(data),
        "sha256": hashlib.sha256(data).hexdigest(), "base64": base64.b64encode(data).decode("ascii")}


def render(kind, payload):
    payload = validate_request(kind, payload)
    work = Path("/work")
    if kind == "latex_compile":
        (work/"main.tex").write_text(payload["source"], encoding="utf-8")
        with (work/"compiler.log").open("wb") as log:
            result = subprocess.run(["pdflatex", "-no-shell-escape", "-interaction=nonstopmode", "-halt-on-error",
                "-file-line-error", "main.tex"], cwd=work, stdout=log, stderr=subprocess.STDOUT, timeout=50)
        if result.returncode:
            return {"ok": False, "error": "latex_compiler_error", "diagnostic": (work/"compiler.log").read_text(errors="replace")[-4000:], "artifacts": []}
        return {"ok": True, "artifacts": [artifact(work/"main.pdf", "application/pdf")], "shell_escape": False}
    if kind == "plot":
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
        plt.rcParams.update({"text.usetex": False, "svg.fonttype": "none", "font.family": "sans-serif",
            "font.sans-serif": ["Noto Sans CJK JP", "DejaVu Sans"]})
        fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
        for series in payload["series"]:
            getattr(ax, {"line": "plot", "scatter": "scatter", "bar": "bar"}[payload["style"]])(
                series["x"], series["y"], label=series["label"])
        ax.set(title=payload.get("title", ""), xlabel=payload.get("xlabel", ""), ylabel=payload.get("ylabel", ""))
        if any(s["label"] for s in payload["series"]):
            ax.legend()
        ax.spines[["top", "right"]].set_visible(False)
        outputs = []
        for fmt in payload["formats"]:
            file = work/("figure." + fmt)
            fig.savefig(file, format=fmt, dpi=payload["dpi"])
            outputs.append(artifact(file, {"png": "image/png", "pdf": "application/pdf", "svg": "image/svg+xml"}[fmt]))
        plt.close(fig)
        return {"ok": True, "artifacts": outputs, "matplotlib_version": matplotlib.__version__, "backend": "Agg"}
    # Trusted deployment self-test; never exposed as a user research tool.
    checks = {"non_root": os.getuid() == 65532, "docker_socket_absent": not Path("/var/run/docker.sock").exists(),
        "no_host_workspace": not Path("/workspace").exists(), "network_unavailable": False, "root_readonly": False}
    try:
        with socket.create_connection(("1.1.1.1", 443), timeout=.5):
            pass
    except OSError:
        checks["network_unavailable"] = True
    try:
        Path("/rsi-readonly-probe").write_text("probe")
    except OSError:
        checks["root_readonly"] = True
    checks["memory_limit"] = Path("/sys/fs/cgroup/memory.max").read_text().strip()
    checks["pids_limit"] = Path("/sys/fs/cgroup/pids.max").read_text().strip()
    checks["cpu_limit"] = Path("/sys/fs/cgroup/cpu.max").read_text().strip()
    checks["file_size_limit"] = resource.getrlimit(resource.RLIMIT_FSIZE)[0]
    return {"ok": all(v is True for v in list(checks.values())[:5]), "checks": checks,
        "packages": Path("/opt/rsi-worker/packages.txt").read_text(), "artifacts": []}


def main():
    resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_ARTIFACT_BYTES, MAX_ARTIFACT_BYTES))
    data = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
    if len(data)>MAX_REQUEST_BYTES:
        raise ValueError("worker request exceeds byte budget")
    request = json.loads(data)
    if request.get("schema") != SCHEMA or set(request) != {"schema", "kind", "payload"}:
        raise ValueError("invalid worker request envelope")
    result = render(request["kind"], request["payload"])
    if sum(a["size"] for a in result["artifacts"]) > MAX_TOTAL_BYTES:
        raise ValueError("total artifact output exceeds budget")
    result.update(schema=SCHEMA, kind=request["kind"])
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        print(json.dumps({"schema": SCHEMA, "ok": False, "error": type(exc).__name__, "artifacts": []}))
        sys.exit(1)
