import base64
import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.runtime import docker_worker as worker
from src.runtime.worker_contract import validate_request, SCHEMA
from src.runtime.research_tools import tool_plan, execute_tool, verify_tool_run
from src.runtime.execution import ExecutionGateway, ExecutionError
from src.util import artifacts_path, sha256_file, write_json, read_json

IMAGE="sha256:"+"1"*64


def test_deployment_profile_preserves_environment_placeholders(tmp_path, monkeypatch):
    import yaml
    from scripts.deploy_docker_worker import write_profile
    from src.util import repo_root

    source = yaml.safe_load((repo_root() / "configs/experiment.yaml").read_text(encoding="utf-8"))
    source["domain_profile"] = "${RSI_TEST_PROFILE_VALUE}"
    source_path = tmp_path / "source.yaml"
    source_path.write_text(yaml.safe_dump(source), encoding="utf-8")
    monkeypatch.setenv("RSI_TEST_PROFILE_VALUE", "private-profile-value")
    output = tmp_path / "profile.yaml"
    write_profile(source_path, output, configured()["execution"]["docker_worker"])
    rendered = output.read_text(encoding="utf-8")
    assert "${RSI_TEST_PROFILE_VALUE}" in rendered
    assert "private-profile-value" not in rendered
    with pytest.raises(FileExistsError):
        write_profile(source_path, output, configured()["execution"]["docker_worker"])


def configured():
    file=artifacts_path("fixture-deployment.json")
    write_json(file,{"schema":"docker-render-deployment-v1","status":"completed","validation":"passed",
        "image":IMAGE,"context":"fixture-local","sources":worker.source_identity()})
    return {"execution":{"enabled":True,"allow_latex":True,"docker_worker":{"enabled":True,
        "image":IMAGE,"context":"fixture-local","receipt_path":str(file),"receipt_sha256":sha256_file(file),"timeout_seconds":3}}}


def inspection():
    return {"Image":IMAGE,"Config":{"User":"65532:65532","Entrypoint":["/usr/bin/python3","/opt/rsi-worker/worker.py"]},"Mounts":[],
        "HostConfig":{"NetworkMode":"none","ReadonlyRootfs":True,"Privileged":False,"Binds":None,"Devices":[],
            "CapAdd":None,"CapDrop":["ALL"],"SecurityOpt":["no-new-privileges=true"],"Memory":512*1024*1024,"MemorySwap":512*1024*1024,
            "NanoCpus":1000000000,"PidsLimit":64,"Ulimits":[{"Name":"fsize","Soft":8388608,"Hard":8388608},{"Name":"nofile","Soft":128,"Hard":128}],
            "Tmpfs":{"/work":"rw,noexec,nosuid,nodev,size=67108864,mode=1777","/tmp":"rw,noexec,nosuid,nodev,size=16777216,mode=1777"}}}


def mock_cli(args,**kwargs):
    if args[:2]==["context","inspect"]: return json.dumps("unix:///var/run/docker.sock")
    if args[:2]==["image","inspect"]:
        return json.dumps([{"Id":IMAGE,"Config":{"Labels":{"rsi.schema":SCHEMA,**{"rsi."+k:v for k,v in worker.source_identity().items()}}}}])
    if args[0]=="create": return "2"*64
    if args[0]=="inspect": return json.dumps([inspection()])
    if args[0]=="rm": return "2"*64
    raise AssertionError(args)


class Process:
    returncode=0
    def __init__(self,args,**kwargs): self.args=args
    def communicate(self,data=None,**kwargs):
        self.request=json.loads(data)
        file=b"%PDF-1.4\nsynthetic unit fixture"
        item={"name":"main.pdf","media_type":"application/pdf","size":len(file),"sha256":hashlib.sha256(file).hexdigest(),"base64":base64.b64encode(file).decode()}
        return json.dumps({"schema":SCHEMA,"kind":self.request["kind"],"ok":True,"artifacts":[item]}).encode(),b""
    def poll(self): return self.returncode


def test_default_denies_renderer_without_host_fallback():
    with patch.object(worker,"_cli") as docker:
        with pytest.raises(ExecutionError,match="Docker"):
            ExecutionGateway({"execution":{"enabled":True}}).run("latex_compile",{"source":"x"})
        docker.assert_not_called()


@pytest.mark.parametrize("payload",[
    {"series":[{"label":"a","x":[1],"y":[float('nan')]}]},
    {"series":[{"label":"a","x":[True],"y":[1]}]},
    {"series":[{"label":"a","x":[1],"y":[1,2]}]},
    {"series":[{"label":"a","x":[1],"y":[1]}],"python":"import os"},
])
def test_bad_plot_inputs_are_rejected_before_docker(payload):
    with patch.object(worker,"_cli") as docker:
        with pytest.raises(ValueError): worker.execute("plot",payload,{})
        docker.assert_not_called()


def test_artifact_receipt_and_cleanup_pass_through_real_public_tool_interface():
    cfg=configured()
    value={"kind":"latex_compile","payload":{"source":"\\documentclass{article}"}}
    plan=tool_plan(value,cfg)
    with patch.object(worker,"_cli",side_effect=mock_cli) as cli,patch.object(worker.subprocess,"Popen",Process):
        result=execute_tool(value,cfg,plan["plan_sha256"])
    inner=result["result"]
    assert inner["ok"] and Path(inner["artifacts"][0]["path"]).read_bytes().startswith(b"%PDF")
    assert read_json(Path(inner["run_dir"])/"cleanup.json")["status"]=="completed"
    assert result["receipt"]["worker_receipt_sha256"]==sha256_file(Path(inner["run_dir"])/"receipt.json")
    assert any(call.args[0][:2]==["rm","--force"] for call in cli.call_args_list)
    assert verify_tool_run(result["run_dir"])["result_ok"]
    Path(inner["artifacts"][0]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError,match="artifact"):
        verify_tool_run(result["run_dir"])


def test_unknown_create_failure_still_cleans_the_exact_generated_name():
    cfg=configured()
    def failing(args,**kwargs):
        if args[0]=="create": raise worker.WorkerUnavailable("unknown create result")
        return mock_cli(args,**kwargs)
    with patch.object(worker,"_cli",side_effect=failing) as cli:
        with pytest.raises(worker.WorkerUnavailable,match="unknown create"):
            worker.execute("latex_compile",{"source":"x"},cfg)
    cleanup=[c.args[0] for c in cli.call_args_list if c.args[0][0]=="rm"]
    assert len(cleanup)==1 and cleanup[0][-1].startswith("rsi-render-")


def test_policy_drift_and_changed_deployment_receipts_fail_closed():
    cfg=configured()
    changed=inspection()
    changed["HostConfig"]["NetworkMode"]="bridge"
    with pytest.raises(worker.WorkerUnavailable,match="policy"):
        worker._inspect_policy(changed,worker.settings(cfg))
    changed=inspection()
    changed["Mounts"]=[{"Source":"host-workspace"}]
    with pytest.raises(worker.WorkerUnavailable,match="policy"):
        worker._inspect_policy(changed,worker.settings(cfg))
    file=Path(cfg["execution"]["docker_worker"]["receipt_path"])
    file.write_text("{}",encoding="utf-8")
    with pytest.raises(worker.WorkerUnavailable,match="receipt"):
        worker.settings(cfg)


def test_context_cannot_redirect_requests_to_a_remote_docker_host():
    cfg=configured()
    with patch.object(worker,"_cli",return_value=json.dumps("tcp://unapproved-host:2375")) as cli:
        with pytest.raises(worker.WorkerUnavailable,match="remote"):
            worker.execute("latex_compile",{"source":"x"},cfg)
        assert cli.call_count==1


def test_timeout_has_a_receipt_and_removes_only_its_owned_container():
    import subprocess
    cfg=configured()
    class TimeoutProcess(Process):
        def communicate(self,data=None,**kwargs):
            if data is not None: raise subprocess.TimeoutExpired(self.args,3)
            return b"",b""
        def kill(self): self.returncode=-9
    value={"kind":"latex_compile","payload":{"source":"x"}}
    plan=tool_plan(value,cfg)
    with patch.object(worker,"_cli",side_effect=mock_cli) as cli,patch.object(worker.subprocess,"Popen",TimeoutProcess):
        result=execute_tool(value,cfg,plan["plan_sha256"])
    assert result["result"]["error"]=="worker_timeout"
    assert verify_tool_run(result["run_dir"])["result_ok"] is False
    removals=[c.args[0] for c in cli.call_args_list if c.args[0][0]=="rm"]
    assert removals==[["rm","--force","2"*64]]


def test_unapproved_output_name_is_not_written_and_container_is_cleaned():
    cfg=configured()
    class BadArtifact(Process):
        def communicate(self,data=None,**kwargs):
            out,err=super().communicate(data,**kwargs)
            value=json.loads(out)
            value["artifacts"][0]["name"]="../../escape.pdf"
            return json.dumps(value).encode(),err
    with patch.object(worker,"_cli",side_effect=mock_cli) as cli,patch.object(worker.subprocess,"Popen",BadArtifact):
        with pytest.raises(worker.WorkerUnavailable,match="name/type"):
            worker.execute("latex_compile",{"source":"x"},cfg)
    assert any(c.args[0]==["rm","--force","2"*64] for c in cli.call_args_list)
