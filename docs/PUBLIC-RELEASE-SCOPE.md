# 公开版本的功能与材料范围

## 功能状态

| 功能 | 状态 | 入口与条件 |
|---|---|---|
| 合成任务、Skill 路由与轨迹 | 支持 | `scripts/smoke_offline.py`；无模型凭据 |
| 固定任务合成闭环 | 支持 | `scripts/smoke_evolution_loop.py`；临时合成题目与 FakeProvider，不产生科研结论 |
| 固定任务基线、候选、评测、审查、发布和回滚 | 支持，真实执行需数据与凭据 | `scripts/rsi.py`；公开 `public-task-input-v1` 使用自备 JSONL 与来源组隔离，不依赖历史 manifest/vendor。历史配置保留原协议分支，不能复用为新协议实验 |
| 用户材料问答、引用与反馈 | 支持 | `scripts/research_assistant.py`；在线执行需要模型凭据及预算 |
| 反馈候选独立评测、审查与发布 | 支持受控晋升 | `benchmark-run` → `promotion-request` → `promotion-review` → `promotion-publish`；精确候选/父版本/收据绑定，独立选择集改善与引用非劣门；合成夹具仅限隔离工程发布 |
| 文献发现、题录核验、摘要问答 | 受限 | 依赖外部公开 API；部分来源可能限流或缺摘要 |
| 轻量计算、静态分析 | 支持 | 受控工具入口；默认不执行任意脚本 |
| 单文件 pdfLaTeX 与结构化绘图 | 可选 | 需按[部署指南](PUBLIC-DOCKER-WORKER.md)在本机部署 Docker worker；本机部署配置不随公开快照提供 |
| `rsi multiseed`、`rsi ablate` | 支持原生冻结计划 | 显式 `--validate-only` / `--execute`、哈希、seed/条件/cohort、一次执行身份和共享请求账本；仅可见分区，禁止 final |
| paper-qa 文档级网关 | 可选安装后支持 | 原生 Doc/Text 记录与本地片段检索，输出接入 `answer-documents`；不启动 PaperQA agent 或隐式 embedding/模型调用 |
| OA 全文获取与问答 | 支持受限来源 | `fetch-fulltext` → `answer-documents`；显式开放许可、受控 HTTPS 来源、字节/时间/字符上限；支持 UTF-8 XML/HTML/plain text，PDF 需可选解析器 |
| 通用 Web 抓取、多文件 TeX、XeLaTeX、任意脚本执行 | 未支持 | 不在当前受控能力范围中 |

## 文件范围

公开快照由 `scripts/build_public_snapshot.py` 从当前工作区的明确清单生成。包含自有源码、主配置、Skill、文档、不依赖私有材料的公开测试、Pi bridge 源码、Docker worker 源码、两个自有基线 release，以及不含原始题目的[聚合证据记录](PUBLIC-EVIDENCE-2026-10-19.json)。依赖未分发实验收据、数据或 vendor 的历史测试保留在研究工作区，不随快照分发。`vendor/`、`data/`、除基线外的 `artifacts/`、`.workbuddy/`、原始 API 响应和本机部署配置均不包含。公开仓库应以该目录创建新的 Git 历史。

项目自有代码采用根目录 MIT 许可证。`THIRD_PARTY.yml` 记录依赖与数据来源；根目录许可证不覆盖第三方项目、数据集、模型响应或日后单独导入的 Skill。公开快照不再分发这些待核材料。安装时下载的依赖仍按自身许可证使用。

## 公开快照命令

```powershell
py -3.12 scripts/build_public_snapshot.py --output outputs/public-release-20261030-v24
py -3.12 scripts/build_public_snapshot.py --verify outputs/public-release-20261030-v24
```

输出目录必须不存在。工具复制文件并写 `PUBLIC_SNAPSHOT_MANIFEST.json`，验证哈希、文件清单、公开文档的相对链接及常见凭据和个人绝对路径。发布时从输出目录建立新的 Git 仓库；不要推送研究仓库的旧 Git 历史，因为旧历史已包含封存题目与个人工作记录。
