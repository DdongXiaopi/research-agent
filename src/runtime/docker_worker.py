"""A local immutable renderer with no host mounts and one-job containers."""
import base64
import json
import os
from pathlib import Path
import re
import subprocess
import time
from uuid import uuid4

from ..util import artifacts_path, read_json, repo_root, sha256_file, sha256_obj, write_json
from .worker_contract import SCHEMA, MAX_REQUEST_BYTES, MAX_OUTPUT_BYTES, MAX_ARTIFACT_BYTES, MAX_TOTAL_BYTES, validate_request


class WorkerUnavailable(RuntimeError):
    pass


def source_identity():
    root = repo_root()
    return {"worker_sha256": sha256_file(root/"tools/docker-worker/worker.py"),
        "contract_sha256": sha256_file(root/"src/runtime/worker_contract.py"),
        "dockerfile_sha256": sha256_file(root/"tools/docker-worker/Dockerfile")}


def docker_env():
    keys = {"PATH", "SYSTEMROOT", "WINDIR", "COMSPEC", "TEMP", "TMP", "USERPROFILE", "HOME", "APPDATA", "LOCALAPPDATA", "PROGRAMFILES", "PATHEXT"}
    return {k:v for k,v in os.environ.items() if k.upper() in keys}


def _cli(args, *, context, timeout=15, missing_ok=False):
    try:
        result = subprocess.run(["docker", "--context", context, *args], capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout, env=docker_env())
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise WorkerUnavailable("local Docker command unavailable: " + type(exc).__name__) from exc
    if result.returncode:
        if missing_ok and "no such container" in result.stderr.casefold():
            return ""
        raise WorkerUnavailable("local Docker command failed: " + str(result.returncode))
    return result.stdout


def local_endpoint(endpoint):
    return isinstance(endpoint,str) and (endpoint.startswith("npipe:////./pipe/") or endpoint.startswith("unix:///"))


def settings(cfg, *, provisioning=False):
    ex = cfg.get("execution") or {}
    value = ex.get("docker_worker") or {}
    if ex.get("enabled") is not True or value.get("enabled") is not True:
        raise WorkerUnavailable("Docker rendering is disabled; no host execution fallback")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", str(value.get("image", ""))):
        raise WorkerUnavailable("rendering requires an immutable local image ID")
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", str(value.get("context", ""))):
        raise WorkerUnavailable("rendering requires an explicit local Docker context")
    receipt_file = Path(value.get("receipt_path", "")).resolve()
    if not receipt_file.is_file() or sha256_file(receipt_file) != value.get("receipt_sha256"):
        raise WorkerUnavailable("rendering deployment receipt changed or is missing")
    receipt = read_json(receipt_file)
    if (receipt.get("schema") != "docker-render-deployment-v1" or receipt.get("status") != "completed"
            or receipt.get("image") != value["image"] or receipt.get("sources") != source_identity()
            or receipt.get("context") != value["context"]):
        raise WorkerUnavailable("rendering deployment/image/source identity mismatch")
    if receipt.get("validation") != "passed" and not provisioning:
        raise WorkerUnavailable("rendering deployment has not passed its real isolation tests")
    for key, default, low, high in (("timeout_seconds", 60, 1, 120), ("memory_mb", 512, 128, 1024), ("pids_limit", 64, 16, 128)):
        actual = value.get(key, default)
        if type(actual) is not int or not low <= actual <= high:
            raise WorkerUnavailable("invalid rendering resource limit: " + key)
    return {**value, "timeout_seconds": value.get("timeout_seconds",60), "memory_mb": value.get("memory_mb",512), "pids_limit": value.get("pids_limit",64)}


def _policy(options, name):
    memory = str(options["memory_mb"]) + "m"
    return ["create", "--name", name, "--interactive", "--pull=never", "--network=none", "--read-only",
        "--cap-drop=ALL", "--security-opt=no-new-privileges=true", "--user=65532:65532",
        "--memory="+memory, "--memory-swap="+memory, "--cpus=1", "--pids-limit="+str(options["pids_limit"]),
        "--ulimit", "fsize=8388608:8388608", "--ulimit", "nofile=128:128",
        "--tmpfs", "/work:rw,noexec,nosuid,nodev,size=67108864,mode=1777",
        "--tmpfs", "/tmp:rw,noexec,nosuid,nodev,size=16777216,mode=1777", options["image"]]


def _inspect_policy(value, options):
    host, config = value["HostConfig"], value["Config"]
    if (value["Image"] != options["image"] or config["User"] != "65532:65532"
            or host["NetworkMode"] != "none" or not host["ReadonlyRootfs"]
            or host["Privileged"] or host.get("Binds") or host.get("Devices")
            or value.get("Mounts") or host.get("CapAdd") or host.get("CapDrop") != ["ALL"]
            or "no-new-privileges=true" not in host.get("SecurityOpt", [])
            or host["Memory"] != options["memory_mb"]*1024*1024 or host["MemorySwap"] != host["Memory"]
            or host["NanoCpus"] != 1000000000 or host["PidsLimit"] != options["pids_limit"]
            or config.get("Entrypoint") != ["/usr/bin/python3","/opt/rsi-worker/worker.py"]
            or {r["Name"]:(r["Soft"],r["Hard"]) for r in host.get("Ulimits",[])} != {"fsize":(8388608,8388608),"nofile":(128,128)}
            or host.get("Tmpfs") != {"/work":"rw,noexec,nosuid,nodev,size=67108864,mode=1777", "/tmp":"rw,noexec,nosuid,nodev,size=16777216,mode=1777"}):
        raise WorkerUnavailable("created container does not match the rendering isolation policy")


def execute(kind, payload, cfg):
    payload = validate_request(kind, payload)
    options = settings(cfg, provisioning=kind=="selftest")
    context = options["context"]
    endpoint = json.loads(_cli(["context", "inspect", context, "--format", "{{json .Endpoints.docker.Host}}"], context=context))
    if not local_endpoint(endpoint):
        raise WorkerUnavailable("remote Docker endpoints are not approved for rendering")
    image = json.loads(_cli(["image", "inspect", options["image"]], context=context))[0]
    expected_labels = {"rsi.schema":SCHEMA, **{"rsi."+k:v for k,v in source_identity().items()}}
    if image["Id"] != options["image"] or any((image["Config"].get("Labels") or {}).get(k) != v for k,v in expected_labels.items()):
        raise WorkerUnavailable("local image does not match reviewed worker sources")
    request = json.dumps({"schema":SCHEMA,"kind":kind,"payload":payload},ensure_ascii=False).encode("utf-8")
    if len(request) > MAX_REQUEST_BYTES:
        raise ValueError("rendering request exceeds byte budget")
    root = artifacts_path("research_tools", "worker-"+uuid4().hex)
    root.mkdir(parents=True,exist_ok=False)
    write_json(root/"request.json",{"schema":SCHEMA,"kind":kind,"payload":payload})
    write_json(root/"status.json",{"status":"started"})
    name = "rsi-render-"+uuid4().hex
    cid, child, cleaned, create_started = None, None, False, False
    started = time.monotonic()
    try:
        create_started = True
        cid = _cli(_policy(options,name),context=context).strip()
        if not re.fullmatch(r"[0-9a-f]{64}",cid):
            raise WorkerUnavailable("invalid created container identity")
        inspection = json.loads(_cli(["inspect",cid],context=context))[0]
        _inspect_policy(inspection,options)
        write_json(root/"container-policy.json",{"image":inspection["Image"],"config":inspection["Config"],"host_config":inspection["HostConfig"],"mounts":inspection["Mounts"]})
        child = subprocess.Popen(["docker","--context",context,"start","--attach","--interactive",cid],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=docker_env())
        try:
            stdout, stderr = child.communicate(request, timeout=options["timeout_seconds"])
        except subprocess.TimeoutExpired:
            _cli(["rm","--force",cid],context=context,missing_ok=True)
            cleaned = True
            child.kill()
            child.communicate(timeout=10)
            result={"ok":False,"kind":kind,"error":"worker_timeout","run_dir":str(root),"artifacts":[],
                "image":options["image"],"network_requests":0,"model_calls":0,
                "elapsed_ms":round((time.monotonic()-started)*1000),"arbitrary_python_execution":False}
            write_json(root/"result.json",result)
            write_json(root/"receipt.json",{"schema":"docker-render-receipt-v1","image":options["image"],"sources":source_identity(),
                "request_sha256":sha256_file(root/"request.json"),"policy_sha256":sha256_file(root/"container-policy.json"),
                "result_sha256":sha256_file(root/"result.json"),"container_id":cid,"exit_code":None,
                "termination":"forced_removal","model_calls":0,"network_requests":0,"host_mounts":False})
            write_json(root/"status.json",{"status":"timeout","timeout_seconds":options["timeout_seconds"]})
            return result
        if len(stdout) > MAX_OUTPUT_BYTES:
            raise WorkerUnavailable("worker exceeded response byte budget")
        try:
            result = json.loads(stdout)
        except (ValueError, UnicodeError) as exc:
            raise WorkerUnavailable("worker did not return a bounded JSON result") from exc
        if not isinstance(result,dict) or result.get("schema") != SCHEMA or type(result.get("ok")) is not bool:
            raise WorkerUnavailable("invalid worker result schema")
        if result.get("kind") not in (None,kind) or (result["ok"] and result.get("kind") != kind):
            raise WorkerUnavailable("worker result belongs to a different request kind")
        if child.returncode and result["ok"]:
            raise WorkerUnavailable("worker exit contradicts its success result")
        items = result.pop("artifacts",None)
        if not isinstance(items,list) or len(items)>3 or (not result["ok"] and items):
            raise WorkerUnavailable("invalid worker output manifest")
        outputs, size = [], 0
        expected = {"latex_compile":{"main.pdf":"application/pdf"}, "plot":{"figure.png":"image/png","figure.pdf":"application/pdf","figure.svg":"image/svg+xml"}, "selftest":{}}[kind]
        seen = set()
        for item in items:
            if not isinstance(item,dict) or item.get("name") not in expected or item["name"] in seen or item.get("media_type") != expected[item["name"]]:
                raise WorkerUnavailable("unapproved artifact name/type")
            seen.add(item["name"])
            data = base64.b64decode(item["base64"],validate=True)
            import hashlib
            if not 0<len(data)<=MAX_ARTIFACT_BYTES or len(data)!=item["size"] or hashlib.sha256(data).hexdigest()!=item["sha256"]:
                raise WorkerUnavailable("rendered artifact integrity/size mismatch")
            size += len(data)
            if size>MAX_TOTAL_BYTES:
                raise WorkerUnavailable("worker total output exceeds budget")
            if (item["name"].endswith(".pdf") and not data.startswith(b"%PDF-")) or (item["name"].endswith(".png") and not data.startswith(b"\x89PNG\r\n\x1a\n")):
                raise WorkerUnavailable("artifact binary signature mismatch")
            file = root/item["name"]
            file.write_bytes(data)
            outputs.append({k:v for k,v in item.items() if k!="base64"} | {"path":str(file)})
        if result["ok"] and kind!="selftest" and not outputs:
            raise WorkerUnavailable("renderer claimed success without outputs")
        required_names = {"main.pdf"} if kind=="latex_compile" else {"figure."+f for f in payload["formats"]} if kind=="plot" else set()
        if result["ok"] and seen!=required_names:
            raise WorkerUnavailable("renderer did not provide exactly the requested formats")
        result.update(kind=kind,run_dir=str(root),artifacts=outputs,image=options["image"],network_requests=0,model_calls=0,
            elapsed_ms=round((time.monotonic()-started)*1000),arbitrary_python_execution=False)
        write_json(root/"result.json",result)
        write_json(root/"receipt.json",{"schema":"docker-render-receipt-v1","image":options["image"],"sources":source_identity(),
            "request_sha256":sha256_file(root/"request.json"),"policy_sha256":sha256_file(root/"container-policy.json"),
            "result_sha256":sha256_file(root/"result.json"),"container_id":cid,"exit_code":child.returncode,
            "model_calls":0,"network_requests":0,"host_mounts":False})
        write_json(root/"status.json",{"status":"completed" if result["ok"] else "rejected"})
        return result
    except BaseException as exc:
        write_json(root/"status.json",{"status":"cancelled" if isinstance(exc,KeyboardInterrupt) else "failed","error_type":type(exc).__name__})
        raise
    finally:
        cleanup_error = None
        if create_started and not cleaned:
            try:
                _cli(["rm","--force",cid or name],context=context,missing_ok=True)
            except WorkerUnavailable as exc:
                cleanup_error = exc
                write_json(root/"cleanup.json",{"status":"failed","container_id":cid,"container_name":name,"error_type":type(exc).__name__})
                write_json(root/"status.json",{"status":"cleanup_failed","container_id":cid,"container_name":name})
        if child and child.poll() is None:
            child.kill()
            child.communicate(timeout=10)
        if cleanup_error:
            raise WorkerUnavailable("rendering container cleanup failed; inspect exact container " + (cid or name)) from cleanup_error
        write_json(root/"cleanup.json",{"status":"completed","container_id":cid,"container_name":name})
