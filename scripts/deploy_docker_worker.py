"""Build an immutable local renderer; publish a profile after real isolation tests."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from urllib.parse import urlparse
from uuid import uuid4

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import yaml
from src.config import load_config, resolve_placeholders, validate_config
from src.runtime import docker_worker as worker
from src.runtime.worker_contract import SCHEMA
from src.util import artifacts_path, read_json, sha256_file, write_json


def write_profile(source_path, output_path, worker_config):
    """Preserve source placeholders instead of persisting resolved environment values."""
    with Path(source_path).open(encoding="utf-8") as stream:
        profile = yaml.safe_load(stream)
    if not isinstance(profile, dict) or not isinstance(profile.get("execution"), dict):
        raise ValueError("source configuration requires an execution mapping")
    profile["execution"].update(enabled=True, docker_worker=worker_config)
    validate_config(resolve_placeholders(profile))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as stream:
        yaml.safe_dump(profile, stream, allow_unicode=True, sort_keys=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-image",default="debian:bookworm-slim")
    parser.add_argument("--apt-mirror",choices=("http://deb.debian.org","http://mirrors.aliyun.com"),default="http://deb.debian.org")
    parser.add_argument("--build-proxy",help="Credential-free host.docker.internal HTTP proxy for image construction only")
    parser.add_argument("--context",default="desktop-linux")
    parser.add_argument("--config",default="configs/experiment.yaml")
    parser.add_argument("--output-config",type=Path,default=Path("configs/experiment.docker-tools.yaml"))
    args=parser.parse_args()
    if args.output_config.exists():
        raise ValueError("output profile already exists; choose a new profile to avoid overwriting deployment identity")
    cfg=load_config(args.config)
    if args.build_proxy:
        proxy=urlparse(args.build_proxy)
        if (proxy.scheme!="http" or proxy.hostname!="host.docker.internal" or not proxy.port or proxy.username or proxy.password
                or proxy.path not in ("","/") or proxy.query or proxy.fragment):
            raise ValueError("build proxy must be a credential-free local Docker host endpoint")
    root=artifacts_path("worker_deployments","deploy-"+uuid4().hex)
    root.mkdir(parents=True)
    write_json(root/"status.json",{"status":"started"})
    try:
        # Only a local socket may see requests; the image build may use public
        # repositories, but the job containers never have a network or mount.
        endpoint=json.loads(worker._cli(["context","inspect",args.context,"--format","{{json .Endpoints.docker.Host}}"],context=args.context))
        if not worker.local_endpoint(endpoint):
            raise ValueError("deployment requires a local Docker context")
        base=json.loads(worker._cli(["image","inspect",args.base_image],context=args.context))[0]
        digests=base.get("RepoDigests") or []
        if not digests:
            raise ValueError("base image must have a registry digest; pull a reviewed public image first")
        base_ref=digests[0]
        context=root/"build-context"
        context.mkdir()
        project=Path(__file__).resolve().parents[1]
        for name in ("Dockerfile","worker.py"):
            shutil.copy2(project/"tools/docker-worker"/name,context/name)
        shutil.copy2(project/"src/runtime/worker_contract.py",context/"worker_contract.py")
        sources=worker.source_identity()
        tag="rsi-render-worker:"+root.name
        labels={"rsi.schema":SCHEMA,**{"rsi."+k:v for k,v in sources.items()}}
        command=["docker","--context",args.context,"build","--pull=false","--build-arg","BASE_IMAGE="+base_ref,
            "--build-arg","APT_MIRROR="+args.apt_mirror,"--tag",tag]
        for k,v in labels.items():
            command.extend(["--label",k+"="+v])
        if args.build_proxy:
            command.extend(["--build-arg","HTTP_PROXY="+args.build_proxy,"--build-arg","HTTPS_PROXY="+args.build_proxy])
        command.append(str(context.resolve()))
        print("building immutable rendering image",flush=True)
        with (root/"build.log").open("w",encoding="utf-8") as log:
            build=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=worker.docker_env(),timeout=1800)
        if build.returncode:
            raise RuntimeError("renderer image build failed; see build.log")
        image=json.loads(worker._cli(["image","inspect",tag],context=args.context))[0]
        receipt={"schema":"docker-render-deployment-v1","status":"completed","image":image["Id"],
            "base_image":base_ref,"base_image_id":base["Id"],"apt_mirror":args.apt_mirror,"apt_signature_verification":True,
            "context":args.context,"sources":sources,
            "build_log_sha256":sha256_file(root/"build.log"),"host_mounts":False,"runtime_network":False,
            "build_network":True,"validation":"pending","verification":{}}
        receipt.update(build_proxy_used=bool(args.build_proxy),proxy_propagated_to_runtime=False)
        receipt_path=root/"receipt.json"
        write_json(receipt_path,receipt)
        cfg["execution"].update(enabled=True,docker_worker={"enabled":True,"context":args.context,"image":image["Id"],
            "receipt_path":str(receipt_path.resolve()),"receipt_sha256":sha256_file(receipt_path),
            "timeout_seconds":60,"memory_mb":512,"pids_limit":64})
        validate_config(cfg)
        print("verifying real container isolation",flush=True)
        probe=worker.execute("selftest",{},cfg)
        if not probe["ok"]:
            raise RuntimeError("renderer isolation self-test failed")
        receipt["validation"]="passed"
        receipt["verification"]["isolation"]=probe
        write_json(receipt_path,receipt)
        cfg["execution"]["docker_worker"]["receipt_sha256"]=sha256_file(receipt_path)
        print("verifying actual PDF and figure outputs",flush=True)
        latex=worker.execute("latex_compile",{"source":r"\documentclass{article}\begin{document}RSI worker smoke: $E=mc^2$.\end{document}"},cfg)
        plot=worker.execute("plot",{"title":"Docker renderer smoke","xlabel":"x","ylabel":"y",
            "series":[{"label":"observations","x":[0,1,2],"y":[0,1,4]}]},cfg)
        if not latex["ok"] or not plot["ok"] or len(plot["artifacts"])!=3:
            raise RuntimeError("rendering smoke did not produce expected artifacts")
        receipt["verification"].update(latex=latex,plot=plot)
        write_json(receipt_path,receipt)
        cfg["execution"]["docker_worker"]["receipt_sha256"]=sha256_file(receipt_path)
        validate_config(cfg)
        write_profile(args.config, args.output_config, cfg["execution"]["docker_worker"])
        write_json(root/"status.json",{"status":"completed","config":str(args.output_config.resolve())})
        print(json.dumps({"status":"completed","image":image["Id"],"receipt":str(receipt_path),"config":str(args.output_config)}))
    except BaseException as exc:
        write_json(root/"status.json",{"status":"cancelled" if isinstance(exc,KeyboardInterrupt) else "failed","error_type":type(exc).__name__})
        if (root/"receipt.json").exists():
            receipt=read_json(root/"receipt.json")
            receipt.update(status="failed",validation="failed")
            write_json(root/"receipt.json",receipt)
        raise


if __name__=="__main__":
    main()
