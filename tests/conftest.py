"""pytest 兼容入口。

本项目当前用 `unittest discover` 跑测试，隔离逻辑在 `tests/__init__.py`
（经 `tests/_sandbox.py`）里完成。若将来改用 pytest，这个文件保证隔离同样生效。
"""

from . import _sandbox  # noqa: F401  — 导入即生效
