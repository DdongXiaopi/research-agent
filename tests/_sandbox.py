"""测试期产物隔离（P1）。

## 解决什么问题

本项目的运行产物就放在仓库里的 `artifacts/`，包括**"当前生效版本"指针**
（`artifacts/releases/ACTIVE`）和基线（`artifacts/releases/baseline-v0/`）。
而部分测试会调用 `init_baseline()` —— 它会顺手 `write_active()`。

后果是：跑一遍测试，真实的"当前生效版本"被拨回出厂设置，基线的时间戳也被改写。
下一次做终评时，系统拿基线**跟它自己**比，Δ 恒为 0，**而且不报错**。
你会以为"这次改进没效果"，其实是被测试踩坏了。

## 对齐开源做法

paper-qa 的 `pqa_directory()`（`src/paperqa/utils.py:523`）与 dspy 的
`default_cache_dir()`（`dspy/utils/caching.py:5`）都是同一个招式：
**用一个唯一的函数读环境变量**，测试时把这个变量指到临时目录。
本项目在 `src/util.py` 里对应的是 `artifacts_path()` / `RSI_HOME`。

paper-qa 特意在**函数内部**读环境变量、而不是在模块顶部读成常量 ——
因为模块级常量一旦求值就固定了，之后改环境变量无效。本项目同理：
部分路径常量（如 `RELEASES_ROOT`）仍在模块级求值，所以 `RSI_HOME`
必须在**任何 src 模块被导入之前**设置好。这正是本模块要做的事。

## 为什么是 `__init__.py` 而不是 conftest.py

本项目用 `python -m unittest discover` 跑测试，pytest 的 `conftest.py`
不会被执行。`tests/__init__.py` 在 discover 导入测试包时最先执行，
早于任何 `src.*` 模块 —— 这个顺序是必需的（见上）。
若将来改用 pytest，`tests/conftest.py` 会复用这里的 `_isolate()`。

## 怎么用

默认开启。如需临时关掉（比如想让测试直接写真实产物），设环境变量：

    RSI_TEST_ISOLATE=0 python -m unittest discover -s tests -t .
"""

from __future__ import annotations

import atexit
import os
import shutil
import tempfile
from pathlib import Path

_SANDBOX: Path | None = None


def _isolate() -> Path | None:
    """Create two explicit baseline fixtures; never copy local experiment history."""
    global _SANDBOX
    if _SANDBOX is not None:
        return _SANDBOX

    if os.environ.get("RSI_HOME"):
        return None          # 外部已指定（例如 CI 或调试），尊重它
    if os.environ.get("RSI_TEST_ISOLATE") == "0":
        return None          # 显式关闭

    tmp = Path(tempfile.mkdtemp(prefix="rsi-artifacts-"))
    os.environ["RSI_HOME"] = str(tmp)
    _SANDBOX = tmp
    atexit.register(lambda: shutil.rmtree(tmp, ignore_errors=True))
    # Bind artifact paths only after isolation. Both fixtures are constructed
    # from checked-in Prompt/Skill sources, not ACTIVE or historical results.
    # The second release supplies an explicit archived/promotion target.
    from src.governance.release import init_baseline, write_active
    init_baseline(release_id="baseline-v0", note="deterministic test baseline")
    init_baseline(release_id="fixture-parent", note="deterministic alias test target")
    write_active("baseline-v0")
    (tmp / "runs").mkdir(parents=True, exist_ok=True)
    return tmp


_isolate()
