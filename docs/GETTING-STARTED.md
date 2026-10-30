# 从公开快照开始

当前验证平台为 Windows 和 CPython 3.12。公开快照不包含科研数据、封存评测题、模型响应、`vendor/` 源码或本机 Docker 部署收据。
若 Windows 的 `py` 启动器未登记 Python 3.12，请把下方 `py -3.12` 换成实际安装的 Python 3.12 可执行文件路径。

## 1. 安装与离线运行

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:RSI_HOME = Join-Path $PWD 'outputs/offline-smoke'
.\.venv\Scripts\python.exe scripts/smoke_offline.py
Remove-Item Env:RSI_HOME
```

成功输出包含 `status: ok` 和 `provider_requests: 0`。这个命令用合成任务和 FakeProvider 检查任务运行、Skill 路由与轨迹，不能证明模型质量或候选增益。公开快照附带 `baseline-v0` 和当前 `baseline-v1` release，`python scripts/rsi.py status` 可查看当前基线。

`python scripts/smoke_public_workflow.py` 进一步在临时产物目录中串起基线初始化、开放任务计划、FakeProvider 回答、引用身份核验、轨迹审计、合成反馈导出和候选计划生成，全程零 API 请求。合成反馈随临时目录销毁；这个检查只验证组件交接，不充当科研效果评估。

`python scripts/smoke_evolution_loop.py` 在一次性源码副本中生成合成题目及三层冻结 split，然后执行固定任务的基线、候选评测、人工拒绝、批准门禁、回滚和泄漏检查。运行结束后副本、题目及产物全部销毁；输出 `status: ok` 和 `provider_requests: 0` 表示固定任务模块已串联，不能推断候选在真实数据上的收益。

## 2. 三个命令入口的关系

| 入口 | 主要输入 | 主要输出 | 用途 |
|---|---|---|---|
| `scripts/research_assistant.py` | 用户提供的材料、文献查询、反馈或开放任务集 | 计划、回答、引用、反馈和开放评测包 | 日常科研任务与开放式评估 |
| `scripts/rsi.py` | 冻结的数据、配置与 release | 基线、候选、门禁、审查请求与 release | 固定任务的进化实验与发布治理 |
| `scripts/ai_grounding.py` | 独立参考与评分计划 | 判分器校准和重算报告 | 独立语义判分校准 |

开放科研任务的反馈可通过 `research_assistant.py feedback` 留在私有运行目录；只有明确标记可用于进化的开发反馈，才能经 `export-dev` 进入候选提案。候选继续经过独立评测和审查，不能由一条反馈自动发布。

## 3. 零模型调用的开放任务计划

示例请求在 `configs/research/request.example.json`。以下命令只验证输入并保存计划：

```powershell
$env:RSI_HOME = Join-Path $PWD 'outputs/research-plan'
.\.venv\Scripts\python.exe scripts/rsi.py init
.\.venv\Scripts\python.exe scripts/research_assistant.py answer --request configs/research/request.example.json
```

`RSI_HOME` 将 release、计划和运行产物写入隔离目录。数据准备另将 manifest、可见任务和封存任务写入当前仓库的 `data/`；公开仓库默认忽略整个 `data/`，只在确认许可和隐私范围后显式选择共享文件。若要让数据准备也与工作区隔离，请在一次性克隆中执行。

实际调用模型时，先在本机 `.env` 或环境变量中设置凭据，选用相应 HTTP 配置，核对返回的计划哈希和费用上限，再显式传 `--execute --expected-plan-sha256 <计划哈希>`。开放科研 CLI 的 `answer`、`discover`、`lookup`、`tool`、`propose-skill`、AI 标注、维护评测和开放基准运行都要求执行时提供匹配哈希；先不带 `--execute` 生成零调用计划。在线请求、数据下载和候选发布仍有各自的预算与审查条件。

全文文档可按 `fetch-fulltext` → `answer-documents` 接入相同计划/回答/引用接口。可选 PDF 解析安装 `requirements-documents.txt`；原生 PaperQA 文档记录安装 `requirements-paperqa.txt`。受控全文来源、公开 task_input、反馈候选晋升、多 seed 与消融命令见[高级能力与验收方案](PUBLIC-CAPABILITY-COMPLETION-2026-10-29.md)。

## 4. 固定任务进化流程

以下是历史 LitQA2 / ProtocolQA 分支的准备方法，需要单独取得许可允许使用的 Parquet 文件。公开新协议可改用上一节链接中的自备 JSONL 流程，无需这两套历史数据。Parquet 文件放在仓库外，按以下顺序生成 manifest、可见任务及封存题，再冻结 split。准备命令统一检查两份输入的必需列、非空题目 ID 和 ID 唯一性，失败时不保留部分新产物；已经冻结的数据集不会被覆盖：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-data.txt
.\.venv\Scripts\python.exe scripts/make_splits.py --litqa2-parquet <LitQA2文件路径> --protocolqa-parquet <ProtocolQA文件路径>
.\.venv\Scripts\python.exe scripts/rsi.py freeze --datasets litqa2 protocolqa
```

此步骤只完成输入准备和冻结。默认 FakeProvider 的固定任务链可按 `rsi init`、`baseline`、`round` 或 `campaign`、`review`、`rollback` 继续；真实模型执行必须遵守 CLI 的冻结计划、请求上限和精确哈希要求。公开 `task_input` 使用新的 `configs/experiment.public-task-input.yaml` 和 JSONL 准备入口，历史研究配置仍保留未分发材料的指纹，不得冒充同一协议。公开快照不附带真实 final 题目，也不提供“开箱即测”的科研增益结论。

`python scripts/smoke_data_preparation.py` 使用一次性合成 Parquet 检查“准备→冻结→计划绑定基线→候选轮次→拒绝审查→回滚”；需先安装 `requirements-data.txt`，运行不读取或改动真实数据。公开 CI 单独运行此项。合成数据上的低正确率仅用于检查接线。

使用自备数据时，先把 `RSI_HOME` 指向新的隔离目录，再初始化 release；基线先生成零调用计划，核对输出的 `plan_sha256` 和请求上限后，才执行同一哈希：

```powershell
$env:RSI_HOME = Join-Path $PWD 'outputs/my-local-run'
.\.venv\Scripts\python.exe scripts/rsi.py init
.\.venv\Scripts\python.exe scripts/rsi.py baseline --datasets litqa2 protocolqa --limit 10 --validate-only
.\.venv\Scripts\python.exe scripts/rsi.py baseline --datasets litqa2 protocolqa --limit 10 --execute --expected-plan-sha256 <上一步的plan_sha256>
.\.venv\Scripts\python.exe scripts/rsi.py round --datasets litqa2 protocolqa --limit 10
```

`round` 返回候选状态及审查请求；仅对实际返回的候选 ID 使用 `review`，可用 `rollback --to baseline-v1` 恢复基线。FakeProvider 的候选可能被拒绝，不能据此推断真实模型效果。

要使用可选 DSPy/GEPA 优化器，先核对 [上游锁定来源](../UPSTREAM_LOCK.yml)及其许可，再运行 `.\.venv\Scripts\python.exe scripts/bootstrap_project.py --restore-vendor --install-optimizers`；该命令会联网下载锁定源码、校验哈希并安装依赖，不能只执行 `pip install -r requirements-mipro.txt`。Pi bridge 可单独用 `.\.venv\Scripts\python.exe scripts/bootstrap_project.py --install-pi` 安装，无需恢复 DSPy/GEPA；也可在 `tools/pi-bridge` 执行 `npm ci`。[Docker 工具部署指南](PUBLIC-DOCKER-WORKER.md)说明如何在目标机器生成和验证本地镜像收据。

公开快照只分发不依赖私有题目、vendor 和历史实验收据的测试；安装 `pytest` 后运行 `python -m pytest -q` 可检查全部随包测试。研究工作区保留完整历史测试及其私有输入，不能把公开测试通过解释为这些历史实验可重跑。

功能状态及再分发边界见 [公开范围](PUBLIC-RELEASE-SCOPE.md)，实验声明见 [公开状态](PUBLIC-STATUS.md)。

公开的历史聚合摘要在 [证据记录](PUBLIC-EVIDENCE-2026-10-19.json)。运行 `python scripts/verify_public_evidence.py` 可零调用检查摘要内部一致性；若合法持有原始本地收据，还可传 `--private-dir <收尾目录>` 核对文件哈希与数值。公开快照不包含该目录。

`python scripts/smoke_public_task_input.py` 可在一次性公开源码副本中准备合成 JSONL、冻结分区，执行新协议的多 seed、消融和三轮 campaign，并检查封存集泄漏。三轮可能没有候选或发布，演练仅验证调度。`python scripts/smoke_open_promotion.py` 检查反馈→Skill 候选→独立评测→批准→发布→回滚及篡改拒绝，全程零 API。真实 HTTP/Pi 验收使用单独的持久隔离目录，并验证免费模型现价；其结果仅证明工程接线。

## 文献来源与工作台

`discover` 支持 `crossref`、`openalex`、`semantic_scholar`、`europe_pmc` 和 `pubmed`；`pubmed` 通过 Europe PMC 的 `SRC:MED` 子集检索 PubMed 记录。可用摘要按 DOI、PMID 和 PMCID 去重后接入回答。启动本地界面：

```powershell
.\.venv\Scripts\python.exe scripts/research_web.py --config configs/experiment.public-task-input.yaml
```

访问 `http://127.0.0.1:8765`。使用 Pi 时，运行 `python scripts/bootstrap_project.py --install-pi`，编辑 `configs/experiment.pi-research.example.yaml` 选择模型和端点，设置配置中的密钥环境变量，再通过 `--config` 指定该文件。命令行六阶段工作流见 `python scripts/research_assistant.py workflow --help`。

要从已授权的反馈创建候选，先使用 `export-dev` 导出开发集，再运行 `propose-candidate --kind system_prompt`、`skill_description`、`skill_body` 或 `new_skill`。已有 Skill 须指定 `--target`；生成需要 HTTP reflection 模型，候选继续经过独立评测和审查。`python scripts/smoke_open_promotion.py --kind skill_body` 可在隔离环境演练完整链路，其他类型可替换 `--kind`。
