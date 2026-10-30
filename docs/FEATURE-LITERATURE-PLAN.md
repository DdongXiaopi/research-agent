# 文献来源整合：选型、接口与实施

分支：`codex/literature-sources`。基于公开 main，独立开发和验收。

## 技术选型与接口

复用现有 `src.retrieval.literature` 计划、预算、请求日志和快照契约；接入 Europe PMC REST core 元数据。PubMed 来源通过 Europe PMC 的 `SRC:MED` 查询子集，保留 PMID、PMCID、DOI 及可用摘要；不把元数据或开放链接当作全文。跨来源以 DOI、PMID、PMCID 去重。请求仍使用现有超时、响应大小、重试和总请求限制。

入口为 `build_plan(discover|lookup, providers=...)`、`execute(plan)` 与 `evidence_from_snapshot(snapshot)`；输出兼容现有 `literature-snapshot-v1`。单源失败记录在事件中，快照摘要进入回答前必须校验内容身份。

## 实施与验收

1. 增加两种来源、规范化字段和身份去重。
2. 单元和集成测试覆盖正常响应、错误来源/DOI、请求预算、快照篡改、摘要传递与跨来源去重。
3. 实际查询一个公开 DOI，验证两个来源及摘要链路；保留请求次数和验收范围。
4. 通过本分支测试后提交；在集成分支验证与 Pi、界面连接，最终更新公开快照及 CI 后发布。
