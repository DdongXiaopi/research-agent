# RSI · 自进化科研助手

**一个面向科研任务的 AI 助手与自进化实验框架。**

RSI（Research Self-Improvement）将文献与用户材料问答、引用核验、反馈学习、Prompt / Skill 优化和评测发布连接成一条可追踪的工作流。你可以用它处理科研问题，也可以围绕自己的数据和评价标准，持续迭代助手的能力。

[![GitHub Actions](https://github.com/DdongXiaopi/research-agent/actions/workflows/offline.yml/badge.svg)](https://github.com/DdongXiaopi/research-agent/actions/workflows/offline.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 项目能做什么

- **科研问答与引用**：使用实验记录、文献摘录等材料回答问题，记录回答、引用片段和执行轨迹，并核验引用与原始材料的对应关系。
- **文献发现与全文阅读**：从 Crossref、OpenAlex、Semantic Scholar、Europe PMC 和 PubMed 发现文献并核验元数据，获取开放获取全文，将文档片段接入问答；支持本地文档检索和 PaperQA 文档接口。
- **Prompt 与 Skill 进化**：从任务运行与反馈中生成候选，优化系统 Prompt、Skill 描述和正文，或提出新的 Skill。
- **评测与实验**：准备自有任务数据，按来源分组并冻结数据分区，运行多 seed 对照、消融和独立开放评测，生成可追踪的报告。
- **审查、发布与回滚**：将候选、评测结果和审查决定绑定到版本，通过门禁后发布，并可回滚到已有版本。
- **科研工具**：提供受控计算与静态分析；通过 Docker worker 生成图表和编译单文件 LaTeX。

## 工作流程

```text
科研问题 / 文献 / 实验记录 / 自有任务数据
                    │
                    ▼
          检索与材料整理 → 问答与引用
                    │
                    ▼
          运行轨迹与反馈 → Prompt / Skill 候选
                    │
                    ▼
            独立评测 → 审查 → 发布 / 回滚
```

每一步都可以通过命令行单独使用，也可以组合为完整的科研助手迭代流程。

## 快速开始

### 1. 安装

克隆仓库并创建 Python 3.12 虚拟环境：

```bash
git clone https://github.com/DdongXiaopi/research-agent.git
cd research-agent
```

**Windows PowerShell**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**Linux / macOS**

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

### 2. 运行一个科研问答

项目提供了一个实验记录示例：[request.example.json](configs/research/request.example.json)。下面以 Windows PowerShell 为例，创建运行目录、生成计划并执行：

```powershell
$python = ".\.venv\Scripts\python.exe"
$env:RSI_HOME = Join-Path $PWD "outputs/my-research"

& $python scripts/rsi.py init
$plan = & $python scripts/research_assistant.py answer --request configs/research/request.example.json | ConvertFrom-Json
& $python scripts/research_assistant.py answer --request configs/research/request.example.json --execute --expected-plan-sha256 $plan.plan_sha256
```

命令会输出回答、引用和运行目录。默认配置使用本地 FakeProvider，适合快速体验；配置 HTTP 模型后，可以用同样的流程处理实际科研材料。

**Linux / macOS**

```bash
export RSI_HOME="$PWD/outputs/my-research"
.venv/bin/python scripts/rsi.py init
plan=$(.venv/bin/python scripts/research_assistant.py answer --request configs/research/request.example.json | .venv/bin/python -c 'import json,sys; print(json.load(sys.stdin)["plan_sha256"])')
.venv/bin/python scripts/research_assistant.py answer --request configs/research/request.example.json --execute --expected-plan-sha256 "$plan"
```

完整配置与在线模型使用方法见[开始使用](docs/GETTING-STARTED.md)。

### 使用科研工作台

初始化后启动本地界面，在浏览器中打开 `http://127.0.0.1:8765`：

```powershell
.\.venv\Scripts\python.exe scripts/research_web.py --config configs/experiment.public-task-input.yaml
```

工作台支持科研问题、材料或文献来源选择，以及回答、引用、六阶段轨迹和反馈查看。安装 Pi bridge 并配置模型后，可使用 `configs/experiment.pi-research.example.yaml`。

### 3. 体验完整工作流

```powershell
# 科研问答 → 引用核验 → 反馈与候选计划
.\.venv\Scripts\python.exe scripts/smoke_public_workflow.py

# 自备任务准备 → 多 seed、消融与三轮进化
.\.venv\Scripts\python.exe scripts/smoke_public_task_input.py

# 反馈 → Skill 候选 → 评测 → 发布与回滚
.\.venv\Scripts\python.exe scripts/smoke_open_promotion.py
```

这些示例使用合成材料，可用于熟悉各模块的输入、输出和衔接方式。

## 常见使用场景

| 场景 | 入口 | 说明 |
|---|---|---|
| 基于材料回答科研问题 | `scripts/research_assistant.py answer` | 输入问题和证据，生成回答、引用与轨迹 |
| 检索和阅读文献 | `discover`、`lookup`、`fetch-fulltext`、`answer-documents` | 从文献发现到全文片段问答 |
| 使用自有任务数据 | `scripts/prepare_public_tasks.py` | 准备 JSONL 任务、来源分组和数据分区 |
| 运行进化实验 | `scripts/rsi.py` | 执行基线、候选轮次、多 seed 对照和消融 |
| 将反馈转为能力 | `feedback`、`export-dev`、`propose-candidate` | 从授权反馈生成待评测的 Prompt 或 Skill 候选 |
| 审查与管理版本 | `promotion-review`、`promotion-publish`、`rollback` | 审查候选、发布版本和回滚 |

查看所有命令：

```powershell
.\.venv\Scripts\python.exe scripts/research_assistant.py --help
.\.venv\Scripts\python.exe scripts/rsi.py --help
```

## 配置与扩展

- **模型与运行时**：通过 `configs/` 选择模型、运行预算和任务配置；支持 HTTP 模型与 Pi bridge。
- **Skill**：在 `skills/` 管理研究流程和路由配置，让不同任务调用相应的知识与操作步骤。
- **文档与数据**：按需安装 `requirements-data.txt`、`requirements-documents.txt` 或 `requirements-paperqa.txt`，接入自有任务与文献材料。
- **工具执行**：按照 [Docker worker 部署指南](docs/PUBLIC-DOCKER-WORKER.md)启用图表与 LaTeX 工具。
- **优化器**：通过 `scripts/bootstrap_project.py --restore-vendor --install-optimizers` 安装锁定的优化器依赖；Pi bridge 可通过 `--install-pi` 单独安装。

## 项目结构

```text
research-agent/
├── configs/       # 模型、实验与请求配置
├── skills/        # Skill 注册、描述与正文
├── src/
│   ├── data/      # 数据准备、分区与冻结
│   ├── retrieval/ # 文献发现与文档检索
│   ├── runtime/   # 任务运行、模型和工具接口
│   ├── evolution/ # 反馈、候选生成与进化
│   ├── eval/      # 基准、评测与实验分析
│   └── governance/ # 门禁、审查、版本与回滚
├── scripts/       # 命令行入口与工作流示例
├── tests/         # 公开测试
└── docs/          # 使用、配置与能力文档
```

## 路线图

- 扩展文献来源、文档导入和多格式材料解析。
- 丰富跨文档证据组织、引用核验和科研任务工作流。
- 增加模型与运行时适配，完善任务配置和运行体验。
- 扩展评测数据、指标和实验报告，方便比较不同版本。
- 探索更直观的交互界面与任务管理方式。

## 文档与贡献

- [开始使用](docs/GETTING-STARTED.md)
- [数据准备、开放评测与进化工作流](docs/PUBLIC-CAPABILITY-COMPLETION-2026-10-29.md)
- [Docker worker 部署](docs/PUBLIC-DOCKER-WORKER.md)

欢迎通过 [Issues](https://github.com/DdongXiaopi/research-agent/issues) 提出想法、报告问题或参与改进。

## 许可证

项目代码采用 [MIT License](LICENSE)。第三方依赖与数据遵循各自的许可证，详见 [THIRD_PARTY.yml](THIRD_PARTY.yml) 和 [第三方声明](THIRD_PARTY_NOTICES.md)。
