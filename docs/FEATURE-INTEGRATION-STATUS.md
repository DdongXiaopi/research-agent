# 四项功能集成状态

四个功能分别在 `codex/literature-sources`、`codex/pi-research-workflow`、`codex/research-frontend`、`codex/feedback-candidates` 开发，集成在 `codex/four-capability-integration`，未合并公开 main。

已验证：文献来源分支真实 DOI 查询及去重；Pi 分支开放问答、会话、无来源及评测边界；前端任务和反馈接口；反馈候选静态门与候选状态构造。集成工作区的 `tests/integration` 为 12 passed，Pi bridge 为 15 passed。公开快照 `public-release-four-integration-v1` 的完整 pytest 为 241 passed、14 skipped、49 subtests passed；缺少可选 Pi 依赖的快照测试跳过项由安装依赖的集成工作区补充验证。

补充验证：四类反馈候选均在隔离环境完成合成数据的开发/选择评测、人工审查模拟、发布、回滚和篡改拒绝，分别记录真实工件类型；无真实模型调用，也不代表模型质量提升。集成代码使用真实 PubMed 摘要及 Pi 模型完成一次受限运行，3 条引用身份核验通过，1 次文献请求及 1 次模型请求。

前端补充验证：在 390×844 浏览器视口检查表单、引用和反馈区，使用 Tab、文本输入及回车提交合成材料任务，任务完成并显示已核验引用。检查结束后恢复浏览器视口。

待完成：更新最终公开快照及远程 CI。公开完整测试结果对应前一个集成快照，新增演练代码须重新验收。测试为工程验证，不构成科研效果提升的证据。
