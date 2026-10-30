# RSI：科研助手的 Prompt 与 Skill 自进化工程

本项目实现固定任务的 Prompt/Skill 候选生成、评测、审查、发布与回滚，并提供用户材料问答、文献发现、引用核验和受控工具。当前候选的科研收益尚未获得足够证据，公开版本不发布“自进化提高科研质量”的性能声明。

## 快速开始

已验证环境为 Windows、CPython 3.12。最小依赖用于零网络、零模型调用的合成任务 smoke：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:RSI_HOME = Join-Path $PWD 'outputs/offline-smoke'
.\.venv\Scripts\python.exe scripts/smoke_offline.py
Remove-Item Env:RSI_HOME
```

成功时输出 `status: ok` 和 `provider_requests: 0`。该检查不代表真实模型效果。完整安装、命令交接和输入准备见[开始使用](docs/GETTING-STARTED.md)。

进一步运行 `.\.venv\Scripts\python.exe scripts/smoke_public_workflow.py`，可以在临时目录验证基线、计划、回答、引用、轨迹、反馈与候选计划的交接；同样不会调用模型 API。

运行 `.\.venv\Scripts\python.exe scripts/smoke_evolution_loop.py` 可在一次性合成数据上检查固定任务的基线、候选评测、审查门禁和回滚；所有题目与产物在临时目录中销毁，不代表真实科研收益。

## 功能入口

| 任务 | 命令 | 当前边界 |
|---|---|---|
| 固定任务进化、门禁、审查和回滚 | `scripts/rsi.py` | 真实评测需要自行准备合规数据与冻结计划 |
| 用户材料问答、文献工具、反馈和开放评测 | `scripts/research_assistant.py` | 在线执行需要显式预算和模型凭据 |
| 独立语义判分校准 | `scripts/ai_grounding.py` | 未达到自动接受资格时仅作诊断 |
| 合成任务接线检查 | `scripts/smoke_offline.py` | 不联网、不读取科研数据 |
| 固定任务合成闭环检查 | `scripts/smoke_evolution_loop.py` | 临时合成数据与 FakeProvider，零 API 请求 |

支持、受限、停用的能力逐项列在[公开范围](docs/PUBLIC-RELEASE-SCOPE.md)。[公开状态](docs/PUBLIC-STATUS.md)说明研究结论与可复核材料的边界。

数据准备与开放科研执行入口的修复见[接口修复计划](docs/PROJECT-REPAIR-PLAN-2026-10-29.md)；公开 task_input、反馈晋升、多 seed/消融与文档全文的实现和验收见[能力补齐记录](docs/PUBLIC-CAPABILITY-COMPLETION-2026-10-29.md)。

历史实验的可分享聚合值和来源哈希在[公开证据记录](docs/PUBLIC-EVIDENCE-2026-10-19.json)中；新能力的合成工程验收摘要在[能力验收记录](docs/PUBLIC-CAPABILITY-EVIDENCE-2026-10-29.json)中。`scripts/verify_public_evidence.py` 提供历史摘要的零调用一致性检查。原始逐题材料和在线运行收据不在公开仓库中。

## 公开仓库来源

原研究工作区及其 Git 历史含封存评测题、实验产物和个人工作记录。公开仓库应由 `scripts/build_public_snapshot.py` 生成的全新目录建立，不应直接推送原仓库。公开快照包括自有源码与两个基线 release，不包括数据集、vendor 源码、个人目录或原始模型响应。生成与校验方法见[公开范围](docs/PUBLIC-RELEASE-SCOPE.md)。

项目自有代码采用 [MIT 许可证](LICENSE)。第三方依赖和数据遵循各自条款；来源、未决项和随附补丁的上游许可见 [THIRD_PARTY.yml](THIRD_PARTY.yml) 与 [第三方声明](THIRD_PARTY_NOTICES.md)。
