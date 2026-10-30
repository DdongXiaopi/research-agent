# 公开能力补齐方案与执行记录（2026-10-29）

目标：补齐公开 task_input、开放反馈候选晋升、受控多 seed/消融、文档检索与合法全文读取，并验证真实 HTTP/Pi 的隔离发布和回滚。验收以具体接口、收据、失败门和实跑记录为准；模型答题或科学收益不能预先保证。

| 阶段 | 解决方案 | 验收 | 状态 |
|---|---|---|---|
| 1 | 新增显式 `public-task-input-v1` 协议。仅绑定公开自有源码与实际使用的运行时；历史配置保持原协议分支。提供带来源/许可的 JSONL 准备入口，按材料来源组隔离数据。 | 全新公开副本无需历史 manifest/vendor 即可准备、冻结、运行；同来源不得跨 split；旧协议不自动转换。 | 已通过公开合成演练与单元检查 |
| 2 | 为反馈候选添加独立评测到审查请求的晋升接口；精确绑定候选、父版本、开发/选择评测、原始回答、标注与指标。复用既有人工审查、不可变 release 和回滚。 | 篡改、缺标注、跨分区污染、指标未过门均拒绝；隔离工程夹具能走通过、批准、发布与回滚。 | 已通过隔离合成演练与篡改检查 |
| 3 | 为 `multiseed`/`ablate` 增加原生冻结计划、显式请求上限、共享账本、一次执行身份及报告。只允许可见分区。 | 计划模式零调用；错哈希/重复执行/final 分区拒绝；三 seed 对照与明确条件消融可运行。 | 已通过公开合成演练；Pi 三 seed 正向对照完成 |
| 4 | 实现 paper-qa 文档适配：复用其文档/文本记录接口，不引入隐式 LLM 或 embedding 请求。新增带许可和来源身份的受限 OA 全文读取，转成现有问答证据。 | 文档→检索→证据→引用链闭合；输入变化、无许可、私网/重定向、超大或无效响应拒绝；网络请求有预算和收据。 | 已通过原生适配测试和 PLOS OA 网络演练 |
| 5 | 从公开副本执行新协议与正向治理闭环，先用合成隔离夹具，再用已授权免费模型分别验证 HTTP/Pi。模型 API 保持计划绑定、请求上限和零价限制。 | 保存成功/失败原始响应、trace、ledger、精确发布/回滚身份；真实结果不包装成科研效果提升。 | 已通过真实 Pi 固定任务与真实 HTTP 开放任务的隔离正向发布/回滚；自动生成 Skill 的效果门未过 |
| 6 | 更新公开文档和 CI，重建独立公开快照并从干净克隆验证。 | 相关回归、公开测试、安装与完整闭环通过；保留实际外部限制，生产发布等待具体工件的用户审核。 | 已通过：干净克隆最小安装、77 通过/9 可选跳过/38 subtests，数据准备、PaperQA 可选 12 项、Pi bridge 15 项及快照复验 |

本机工程验收仅使用当时核验为免费的模型；生产候选发布仍须具体工件的用户审查。多文件 TeX、XeLaTeX、任意脚本不是这四组缺口的解决前提，继续遵守既有受控工具范围。

## 使用公开 task_input

输入为外部 JSONL，每行包含 `task_id`、`question`、唯一 `options` 数组、整数 `correct_index`、非空 `evidence` 及可选 `sources` / `group_id` / `domain`。准备程序把共享 evidence、来源或 group_id 的题目归入同一联通组；分组时归一化证据中的 Unicode 和空白，并去除来源标识、group_id 的首尾空白。不同写法的 DOI、URL 或其他来源别名不会自动识别，输入前应统一，或对同一材料显式提供相同 `group_id`。需要足够独立来源组填满三层分区。数据集名以 `public_` 开头，避免碰到历史专用 loader。

准备后的数据写入仓库内 `data/`，`RSI_HOME` 只隔离计划、release 与执行产物；公开仓库默认忽略 `data/`。自备材料若需单独隔离，请在一次性克隆中准备，不要直接推送包含数据的研究工作区。

```powershell
$env:RSI_HOME = Join-Path $PWD 'outputs/public-task-run'
python scripts/prepare_public_tasks.py --input <自备题目.jsonl> --dataset public_example --source <来源版本> --license <明确许可>
python scripts/rsi.py --config configs/experiment.public-task-input.yaml freeze --datasets public_example
python scripts/rsi.py init --task-input-baseline-v2
python scripts/rsi.py --config configs/experiment.public-task-input.yaml baseline --datasets public_example --limit 10 --validate-only
python scripts/rsi.py --config configs/experiment.public-task-input.yaml baseline --datasets public_example --limit 10 --execute --expected-plan-sha256 <上一步哈希>
```

新协议不修改旧 manifest，不复用旧实验分数；HTTP 运行将该公开配置中的 task/runtime provider 改为 http，配置模型和 api_key_env，Pi 改为 pi 并先安装锁定 bridge。实际使用的模型、数据、源码与运行时进入新身份。

## 多 seed 与消融

```powershell
python scripts/rsi.py --config configs/experiment.public-task-input.yaml multiseed --datasets public_example --seeds 11 12 13 --limit 10 --validate-only
python scripts/rsi.py --config configs/experiment.public-task-input.yaml multiseed --datasets public_example --seeds 11 12 13 --limit 10 --execute --expected-plan-sha256 <计划哈希>
python scripts/rsi.py --config configs/experiment.public-task-input.yaml ablate --datasets public_example --seeds 11 12 13 --limit 10 --base-release <基线> --condition A0=<基线> --condition A1=<已评审版本> --validate-only
```

消融执行时使用完全相同参数加 `--execute --expected-plan-sha256 <哈希>`。亦支持明确的 `--prompt-release`、`--desc-release`、`--body-release` 因子组合；description 因子不带入正文变化，正文因子保留基线描述。未提供的因子标记不可估。多 seed 可用 `--compare <release>` 或 `--candidate <未发布候选.json>` 做配对对照。真实 provider 的计划和执行都须显式提供 `--logical-request-cap`、`--physical-request-cap`；失败的执行身份也不可重放。远端 seed 未必控制模型 RNG，报告保留该限制。

## 反馈到发布

先经 `feedback --use-for-evolution` → `export-dev` → `propose-skill` 得到待评测候选。独立开放数据须先冻结 task rubric、来源组与三层分区。开发/选择回答都使用同一 `--candidate` 运行 `benchmark-run`，再经独立双评和必要裁决完成 `benchmark-assess`。

开放 benchmark 的来源文本和 `group_id` 会在跨分区检查时统一 Unicode 与空白形式；同一分区内等价的 `group_id` 会按同一来源组保存，同一来源不得伪装成多个来源组。反馈材料与独立评测也采用相同的来源身份检查。不同 DOI/URL 写法或不同内容片段不一定能自动识别为同一来源，数据提供者仍须为它们指定同一个规范 `group_id`。冻结先在同一文件系统的临时目录生成完整内容，再发布目录；异常会清理临时目录，强制终止可能遗留可清理的临时目录，但不会阻止重新冻结。断电后的持久性仍取决于文件系统。

新冻结记录将上述分组检查策略纳入数据身份；旧冻结记录仍可核验和审计，但未记录该策略的旧材料需要用原始输入重新冻结后才能生成新的评测计划，旧结果不自动迁移。

```powershell
python scripts/research_assistant.py promotion-request --candidate <candidate.json> --config <相同配置> --development-id <open开发ID> --selection-id <open选择ID>
python scripts/research_assistant.py promotion-review --candidate-id <ID> --expected-request-sha256 <精确请求哈希> --decision approve --reviewer <实际审核者> --reason <具体理由>
python scripts/research_assistant.py promotion-publish --candidate-id <ID> --expected-request-sha256 <同一请求哈希>
python scripts/rsi.py rollback --to <父版本>
```

开发与选择必须来自同一冻结 benchmark、精确候选和当前父版本，且与反馈来源无交集。要求开发/选择 rubric 不退步、引用 precision 与覆盖不退步；选择集至少三来源组且配对组 bootstrap 的 rubric 改善区间下界大于零。每次审查和发布重新核对原始回答、trace、标注及文件哈希；缺失指标不补零。任何科学泛化声明仍不自动授权。合成数据和 `engineering_fixture` 标注只接受显式隔离工程模式，不能授权生产发布。

## 全文文档与 PaperQA

```powershell
python scripts/research_assistant.py fetch-fulltext --snapshot <文献snapshot.json> --source-id <来源ID>
python scripts/research_assistant.py fetch-fulltext --snapshot <同一snapshot.json> --source-id <同一来源ID> --execute --expected-plan-sha256 <计划哈希>
python scripts/research_assistant.py answer-documents --request <evidence为空的请求.json> --snapshot <全文snapshot.json> --document-backend local
```

完整问答执行沿用 `--execute --expected-plan-sha256`。可选安装 `requirements-paperqa.txt` 后将 `--document-backend` 改为 `paperqa`；它实际构造上游 Doc/Text 记录再做本地片段检索，不启动其 agent/embedding/自动模型请求。PaperQA 版本固定为 2026.8.12，官方 wheel 身份与许可记录在 `UPSTREAM_LOCK.yml`。PDF 解析器独立在有超时的子进程运行，也可只安装轻量 `requirements-documents.txt`。

全文来源必须已有明确支持的开放许可与 OA URL；当前受控域为 PMC/NCBI、Europe PMC/EBI、arXiv、PLOS、eLife。每次最多四次 GET（含重定向），每跳均校验 HTTPS/域名，最多 8 MB、500000 字符和明确 deadline。下载、解析和检索都保存哈希与收据，失败不伪造正文。通用 Web 抓取及未声明许可的文章不属于该接口。

## 验收入口

- `python scripts/smoke_public_task_input.py`：新公开协议、三 seed 多 seed/消融和三轮 campaign 调度及封存集泄漏检查，零模型调用。
- `python scripts/smoke_open_promotion.py`：合成反馈候选的正向发布、回滚和篡改拒绝，零 API。
- `python scripts/smoke_open_promotion.py --live --run-home <新隔离目录>`：免费 HTTP 模型生成 Skill 提案并做开放独立评测；程序评分明确标为工程夹具，候选可能被效度门拒绝。
- `python scripts/smoke_open_promotion.py --live --authored-candidate --run-home <新隔离目录>`：反馈输入和提案仍走相同接口，提案内容明确标记为人工编写的工程夹具；独立评测回答使用真实 HTTP 模型，检查正向治理接线。
- `python scripts/smoke_public_task_input.py --provider pi --run-home <新隔离目录>`：真实 Pi 固定任务的三 seed 对照、收据核验、隔离批准/发布与回滚；`--provider http` 可验证 HTTP 同一路径。

在线脚本先从官方价格页读取并保存模型 prompt/completion 零价证据，只允许 `Qwen/Qwen3-8B`、`Qwen/Qwen2.5-7B-Instruct` 或 `THUDM/GLM-4-9B-0414` 中当前核验为零价的模型，不回退付费模型；所有调用有原始输出和请求上限。合成码表基线、人工工程提案与程序参考仅用于接线验收，不证明科研收益。

## 已保存的在线证据

- `outputs/public-pi-fixed-acceptance-20261029-v3`：全新公开源副本，真实 Pi/免费 Qwen2.5-7B，12 次逻辑与 12 次物理请求全部 200，三 seed × 开发/选择两分区的基线准确率均为 0、候选均为 1；工程夹具经 `auto-sim` 精确审查、发布并回滚，final 未读取。选择集只保存聚合收据，开发集逐题 trace 可审计。该运行不构成科研收益实验。
- `outputs/public-http-open-authored-20261029-v6`：真实 HTTP/免费 Qwen2.5-7B，12 次请求无未知账本；人工编写并标记来源的反馈 Skill 提案，在合成独立开发/选择集上程序评分均为基线 0、候选 1，引用分均为 1；精确审查、隔离发布、回滚与篡改拒绝通过。程序评分和合成码表不证明真实科研效用。
- `outputs/public-http-open-generated-20261029-v1`：真实 HTTP 自动生成提案和 12 次评测请求完成；提案通过静态门，但未保留反馈中的码表映射，开发/选择 rubric 均为 0→0，独立效果门拒绝发布。这是模型提案效度未过，而非软件缺入口。
- `outputs/public-pi-fixed-acceptance-20261029-v1`：24 次真实请求全部完成，但基线与候选准确率均为 1，改善门正确拒绝；保留失败测量。
- `outputs/public-pi-fixed-acceptance-20261029-v2`：12 组运行发现工程核验错误要求选择集逐题 trace；协议本身不公开该 trace，已按分区修复后在 v3 正向验收。
- `outputs/public-http-fixed-acceptance-20261029-v1`：第 9 个请求遇上游读取超时，一次性计划停在失败状态，账本保留；未重放或声明通过。
- `outputs/public-fulltext-acceptance-20261029-v2/summary.json`：PLOS CC-BY 全文 JATS 网络下载，经真实 PaperQA `Doc`/`Text`、本地片段检索、回答引用身份审计通过；两次下载请求，零模型请求。

上述 `outputs/` 为本机隔离验收材料，不随公开快照分发；公开版本附[聚合摘要及哈希](PUBLIC-CAPABILITY-EVIDENCE-2026-10-29.json)，不把缺失的原始响应称作可独立复现的结果。生产候选仍需要真实来源、独立判分与具体工件的用户审核。
