"""通用工具：路径、哈希、JSON/JSONL 读写、时间。

仅依赖标准库，供运行时与脚本共用。
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List

REPO_ROOT = Path(__file__).resolve().parents[1]

# 产物根目录的环境变量开关（P1）。
#
# 为什么需要：本项目的产物（release 指针、基线、候选、评测结果）就放在仓库里的
# `artifacts/`，而单元测试会调用 `init_baseline()` 之类的函数 —— 它会写
# `artifacts/releases/ACTIVE`。结果就是**跑一遍测试，真实的"当前生效版本"被拨回出厂
# 设置**，终评于是拿基线跟它自己比、Δ 恒为 0 且不报错。
#
# 对齐开源做法：paper-qa 的 `pqa_directory()`（`src/paperqa/utils.py:523`）与 dspy 的
# `default_cache_dir()`（`dspy/utils/caching.py:5`）都是**用一个唯一的函数读环境变量**，
# 测试时把这个变量指到临时目录。这里同构：`RSI_HOME` 指向哪里，产物就写到哪里。
#
# 注意：**在函数内部读**而不是在模块顶部读成常量。模块级常量一旦求值就固定了，
# 之后改环境变量无效（这正是 paper-qa 刻意避开 module-level 的原因）。
ARTIFACTS_ENV_VAR = "RSI_HOME"


def repo_root() -> Path:
    return REPO_ROOT


def artifacts_root() -> Path:
    """产物根目录。优先读 `RSI_HOME`，未设置时用仓库内的 `artifacts/`。"""
    home = os.environ.get(ARTIFACTS_ENV_VAR)
    if home:
        return Path(home).resolve()
    return REPO_ROOT / "artifacts"


def artifacts_path(*parts: str) -> Path:
    """产物目录下的路径。**所有**读写产物的地方都必须走这个函数，不要自己拼。"""
    return artifacts_root().joinpath(*parts)


def resolve_artifact(p: os.PathLike | str) -> Path:
    """把产物路径解析到**当前**产物根（P1）。

    历史产物里存的 `body_path` 是相对仓库根的字符串，形如
    `artifacts/releases/rel-xxx/skills/y.md`。直接 `Path(p).exists()` 会按
    当前工作目录解析 —— 一旦 `RSI_HOME` 把产物重定向到别处（测试沙箱、CI），
    就会查错地方。统一走这个函数：以 `artifacts/` 开头的相对路径改按
    `artifacts_root()` 解析，绝对路径原样返回。
    """
    q = Path(p)
    if q.is_absolute():
        return q
    parts = q.parts
    if parts and parts[0] == "artifacts":
        return artifacts_path(*parts[1:])
    return REPO_ROOT / q


def path(*parts: str) -> Path:
    """仓库内的路径（源码资源：configs/、data/、vendor/ 等）。

    与 `artifacts_path` 的分工：`path` 用于**读**仓库自带的资源，不受 `RSI_HOME`
    影响；`artifacts_path` 用于**读写**运行产物，可被 `RSI_HOME` 整体重定向。
    """
    return REPO_ROOT.joinpath(*parts)


def ensure_dir(p: os.PathLike | str) -> Path:
    q = Path(p)
    q.mkdir(parents=True, exist_ok=True)
    return q


def stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_obj(obj: Any) -> str:
    return sha256_text(stable_json(obj))


def sha256_file(p: os.PathLike | str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def short_hash(text: str, n: int = 12) -> str:
    return sha256_text(text)[:n]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(p: os.PathLike | str, obj: Any) -> Path:
    q = Path(p)
    ensure_dir(q.parent)
    text = json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                dir=q.parent, prefix=q.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        # Antivirus/indexing tools can briefly hold a just-written file on
        # Windows. Keep the already-fsynced temporary and retry only this
        # transient sharing violation; never retry other filesystem errors.
        for attempt in range(6):
            try:
                os.replace(temporary, q)
                break
            except PermissionError:
                if attempt == 5:
                    raise
                time.sleep(0.025 * (2 ** attempt))
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
    return q


def read_json(p: os.PathLike | str) -> Any:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def write_jsonl(p: os.PathLike | str, rows: Iterable[Dict[str, Any]]) -> Path:
    q = Path(p)
    ensure_dir(q.parent)
    with q.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    return q


def commit_new_files(entries: Iterable[tuple[os.PathLike | str, bytes]]) -> None:
    """Stage bytes and create every destination exclusively, rolling back on failure."""
    items = [(Path(path), data) for path, data in entries]
    if not items or len({path.resolve() for path, _ in items}) != len(items):
        raise ValueError("new file transaction requires distinct destinations")
    if any(not isinstance(data, bytes) for _, data in items):
        raise ValueError("new file transaction requires bytes")
    if any(path.exists() or path.is_symlink() for path, _ in items):
        raise FileExistsError("new file transaction refuses to replace existing outputs")
    staged: list[tuple[Path, Path]] = []
    installed: list[tuple[Path, Path]] = []
    try:
        for destination, data in items:
            ensure_dir(destination.parent)
            with tempfile.NamedTemporaryFile(mode="wb", dir=destination.parent,
                    prefix=destination.name + ".", suffix=".tmp", delete=False) as stream:
                pending = Path(stream.name)
                staged.append((pending, destination))
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
        for pending, destination in staged:
            os.link(pending, destination)
            installed.append((pending, destination))
    except BaseException:
        for pending, destination in reversed(installed):
            if destination.exists() and os.path.samefile(pending, destination):
                destination.unlink()
        raise
    finally:
        for pending, _ in staged:
            pending.unlink(missing_ok=True)


def jsonl_bytes(rows: Iterable[Dict[str, Any]]) -> bytes:
    return "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows).encode("utf-8")


def read_jsonl(p: os.PathLike | str) -> List[Dict[str, Any]]:
    q = Path(p)
    if not q.exists():
        return []
    with q.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(p: os.PathLike | str, row: Dict[str, Any]) -> Path:
    q = Path(p)
    ensure_dir(q.parent)
    with q.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return q


def iter_jsonl(p: os.PathLike | str) -> Iterator[Dict[str, Any]]:
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def redact(text: str, extra_patterns: Iterable[str] = ()) -> str:
    """写盘前脱敏：密钥 / Authorization / Cookie / Bearer token。"""
    import re

    patterns = [
        r"(?i)(api[_-]?key\s*[=:]\s*)([A-Za-z0-9_\-]{8,})",
        r"(?i)(authorization\s*:\s*)(bearer\s+)?([A-Za-z0-9_\-\.=]{8,})",
        r"(?i)(cookie\s*:\s*)([^\n]+)",
        r"(?i)(token\s*[=:]\s*)([A-Za-z0-9_\-\.]{8,})",
        r"(sk-[A-Za-z0-9_\-]{16,})",
    ]
    out = text
    for pat in list(patterns) + list(extra_patterns):
        out = re.sub(pat, _mask, out)
    return out


def _mask(m) -> str:
    # 保留前缀（若存在），其余打码
    groups = m.groups()
    if len(groups) >= 2 and groups[0]:
        return f"{groups[0]}***REDACTED***"
    return "***REDACTED***"
