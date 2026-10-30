"""产物隔离守卫（P1）—— 在任何 `src.*` 导入之前兜底触发沙箱。

## 为什么需要它

正确的跑法是 `python -m unittest discover -s tests -t .`（`-t .` 让 `tests` 作为包被导入，
于是 `tests/__init__.py` → `_sandbox` 生效）。

但如果**漏掉 `-t .`**，unittest 会把 `tests/` 当作顶层目录，测试模块以
`unit.test_x` / `e2e.test_x` 这类**顶层包名**导入 —— `tests/__init__.py` 根本不执行，
沙箱不生效，测试会直接改写仓库里真实的 `artifacts/`。

2026-09-26 实测踩到：`discover -s tests`（无 `-t .`）跑完整套后，真实的
`artifacts/releases/ACTIVE` 被 `init_baseline()` + `rollback()` 拨回 `baseline-v0`，
ACTIVE 候选 `rel-cand-34bfab62fe04` 丢失，且**没有任何报错** ——
下次终评会拿基线与它自己比，Δ 恒为 0。

## 这里做什么

在任何 `src.*` 导入之前 `import tests`（触发 `_sandbox`）。
若连 `tests` 包都导入不到（cwd 不在仓库根），宁可让测试收集失败，
也不要静默跑坏真实产物 —— 对齐 gepa api.py:286 的 fail-loud。
"""

from tests import _sandbox  # noqa: F401  — 导入即生效
