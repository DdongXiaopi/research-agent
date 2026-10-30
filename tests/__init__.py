"""测试包初始化：在导入任何 `src.*` 模块之前完成产物隔离。

详见 `tests/_sandbox.py`。必须放在 `__init__.py`（而不是 conftest.py），
因为本项目用 `unittest discover` 跑测试，且部分路径常量在模块级求值。
"""

from . import _sandbox  # noqa: F401  — 导入即生效
