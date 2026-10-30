"""受控执行网关（M3 / Spec FR-004、FR-007）。

它解决的是"模型想算个数、想渲个公式"这类需求，但**不给 shell**：

- Pi 内置 bash / powershell / edit / write 一律不激活（配置层声明，运行层校验）
- 代码执行走 **AST 白名单解释器**：只认常量、四则运算、比较、布尔运算和
  一份短小的函数白名单；`import`、属性访问、赋值、lambda、推导式全部拒绝
- LaTeX 模板保留离线渲染；显式部署的 Docker worker 提供真实单文件 PDF 编译及数据图表
- 输出有长度上限，表达式有长度与节点数上限

必须如实说明的两个限制：
1. **没有真正的超时**。Windows 上没有 `signal.alarm`，这里用"表达式长度 + AST 节点数"
   做静态护栏（白名单里也不存在循环与递归，因此不存在死循环）。
   要接真实代码执行时，必须换成子进程沙箱 + wall-clock 超时，本模块不能充当安全边界。
2. **不是沙箱**。AST 白名单能挡住语法层面的越权，但它跑在同一进程里；
   真实编译和绘图按 Spec 走每任务新建的 Docker 隔离容器，未部署时显式拒绝；任意 Python 仍拒绝。
"""

from __future__ import annotations

import ast
import math
import re
import time
from string import Template
from typing import Any, Dict, List, Optional

from ..util import sha256_text

MAX_EXPR_CHARS = 400
MAX_NODES = 200
MAX_OUTPUT_CHARS = 2000
MAX_INTEGER_BITS = 4096
MAX_CONTAINER_ITEMS = 200

ALLOWED_FUNCS = {
    "abs": abs, "round": round, "min": min, "max": max, "len": len, "sum": sum,
    "int": int, "float": float, "str": str, "bool": bool,
    "log": math.log, "log10": math.log10, "exp": math.exp, "sqrt": math.sqrt,
    "pow": math.pow, "floor": math.floor, "ceil": math.ceil,
}

ALLOWED_NAMES = {"pi": math.pi, "e": math.e, "true": True, "false": False, "null": None}

_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)
_ALLOWED_CMP = (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)
_ALLOWED_UNARY = (ast.UAdd, ast.USub, ast.Not)


class ExecutionError(RuntimeError):
    pass


class WSL2Unavailable(ExecutionError):
    """保留兼容异常名：所请求的隔离能力不可用。**不静默降级**。"""


def _count_nodes(tree: ast.AST) -> int:
    return sum(1 for _ in ast.walk(tree))


def _bounded_value(value: Any, depth: int = 0) -> Any:
    if depth > 10:
        raise ExecutionError("值嵌套过深")
    if isinstance(value, int) and value.bit_length() > MAX_INTEGER_BITS:
        raise ExecutionError("整数超过计算预算")
    if isinstance(value, float) and not math.isfinite(value):
        raise ExecutionError("不允许非有限数值")
    if isinstance(value, str) and len(value) > MAX_OUTPUT_CHARS:
        raise ExecutionError("字符串超过计算预算")
    if isinstance(value, (list, tuple, set, dict)):
        if len(value) > MAX_CONTAINER_ITEMS:
            raise ExecutionError("容器超过计算预算")
        for item in (list(value.items()) if isinstance(value, dict) else value):
            _bounded_value(item, depth + 1)
    elif not isinstance(value, (int, float, str, bool, type(None))):
        raise ExecutionError("计算值类型不受支持")
    return value


def _eval_node(node: ast.AST, variables: Dict[str, Any]) -> Any:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body, variables)
    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float, str, bool, type(None))):
            raise ExecutionError(f"不允许的常量类型: {type(node.value).__name__}")
        return _bounded_value(node.value)
    if isinstance(node, ast.BinOp) and isinstance(node.op, _ALLOWED_BINOPS):
        left, right = _eval_node(node.left, variables), _eval_node(node.right, variables)
        if isinstance(node.op, ast.Mult):
            sequence, count = (left, right) if isinstance(right, int) else (right, left)
            if isinstance(sequence, (str, list, tuple)) and isinstance(count, int):
                maximum = MAX_OUTPUT_CHARS if isinstance(sequence, str) else MAX_CONTAINER_ITEMS
                if len(sequence) * max(0, count) > maximum:
                    raise ExecutionError("重复结果超过计算预算")
        if isinstance(node.op, ast.Pow):
            if not isinstance(right, (int, float)) or abs(right) > MAX_INTEGER_BITS:
                raise ExecutionError("指数超过计算预算")
            if isinstance(left, int) and isinstance(right, int) and right > 0:
                if max(1, abs(left).bit_length() - 1) * right > MAX_INTEGER_BITS:
                    raise ExecutionError("幂结果超过整数预算")
        return _bounded_value(_ALLOWED_BINOPS_OPS[type(node.op)](left, right))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, _ALLOWED_UNARY):
        v = _eval_node(node.operand, variables)
        if isinstance(node.op, ast.UAdd):
            return +v
        if isinstance(node.op, ast.USub):
            return -v
        return not v
    if isinstance(node, ast.Compare):
        left = _eval_node(node.left, variables)
        for op, comparator in zip(node.ops, node.comparators):
            if not isinstance(op, _ALLOWED_CMP):
                raise ExecutionError(f"不允许的比较运算符: {type(op).__name__}")
            right = _eval_node(comparator, variables)
            ok = _ALLOWED_CMP_OPS[type(op)](left, right)
            if not ok:
                return False
            left = right
        return True
    if isinstance(node, ast.BoolOp):
        value = None
        for expression in node.values:
            value = _eval_node(expression, variables)
            if isinstance(node.op, ast.And) and not value:
                return value
            if isinstance(node.op, ast.Or) and value:
                return value
        return value
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        vals = [_eval_node(e, variables) for e in node.elts]
        if isinstance(node, ast.Tuple):
            return tuple(vals)
        if isinstance(node, ast.Set):
            return set(vals)
        return vals
    if isinstance(node, ast.Dict):
        return {_eval_node(k, variables): _eval_node(v, variables)
                for k, v in zip(node.keys, node.values)}
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ExecutionError("只允许调用白名单里的具名函数")
        fn = ALLOWED_FUNCS.get(node.func.id)
        if fn is None:
            raise ExecutionError(f"函数不在白名单: {node.func.id}")
        if node.keywords:
            raise ExecutionError("不允许关键字参数")
        return _bounded_value(fn(*[_eval_node(a, variables) for a in node.args]))
    if isinstance(node, ast.Name):
        if node.id in variables:
            return variables[node.id]
        if node.id in ALLOWED_NAMES:
            return ALLOWED_NAMES[node.id]
        raise ExecutionError(f"未定义的名字: {node.id}")
    raise ExecutionError(f"不允许的语法: {type(node).__name__}")


_ALLOWED_BINOPS_OPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a ** b,
}

_ALLOWED_CMP_OPS = {
    ast.Eq: lambda a, b: a == b,
    ast.NotEq: lambda a, b: a != b,
    ast.Lt: lambda a, b: a < b,
    ast.LtE: lambda a, b: a <= b,
    ast.Gt: lambda a, b: a > b,
    ast.GtE: lambda a, b: a >= b,
}

_BANNED_TOKENS = ("import ", "open(", "__", "exec(", "eval(", "globals", "locals",
                  "subprocess", "os.", "sys.")


def safe_eval(expr: str, variables: Optional[Dict[str, Any]] = None,
              max_chars: int = MAX_EXPR_CHARS, max_nodes: int = MAX_NODES) -> Any:
    """在 AST 白名单内求值；任何越界语法都抛 `ExecutionError`。"""
    expr = (expr or "").strip()
    if not expr:
        raise ExecutionError("空表达式")
    if len(expr) > max_chars:
        raise ExecutionError(f"表达式过长: {len(expr)}/{max_chars}")
    low = expr.lower()
    for t in _BANNED_TOKENS:
        if t in low:
            raise ExecutionError(f"包含被禁止的片段: {t!r}")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ExecutionError(f"语法错误: {exc}") from exc
    n = _count_nodes(tree)
    if n > max_nodes:
        raise ExecutionError(f"表达式过于复杂: {n}/{max_nodes} 个 AST 节点")
    variables = dict(variables or {})
    _bounded_value(variables)
    try:
        return _bounded_value(_eval_node(tree, variables))
    except (ValueError, OverflowError, TypeError) as exc:
        raise ExecutionError(f"计算失败: {type(exc).__name__}") from exc


_PLACEHOLDER_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


def render_latex(template: str, values: Dict[str, Any],
                 max_output_chars: int = MAX_OUTPUT_CHARS) -> str:
    """渲染 LaTeX 模板（`{name}` 占位符），**不编译**。

    未提供的占位符会报错而不是原样留下：留着 `{x}` 会让下游看到半成品公式。
    """
    if len(template or "") > max_output_chars:
        raise ExecutionError(f"模板过长: {len(template)}/{max_output_chars}")
    missing = [k for k in _PLACEHOLDER_RE.findall(template or "") if k not in values]
    if missing:
        raise ExecutionError(f"模板占位符缺少取值: {sorted(set(missing))}")
    out = _PLACEHOLDER_RE.sub(lambda m: str(values[m.group(1)]), template or "")
    if len(out) > max_output_chars:
        raise ExecutionError(f"渲染结果过长: {len(out)}/{max_output_chars}")
    return out


class ExecutionGateway:
    """受控执行入口：轻量表达式 / LaTeX 渲染 /（高风险）WSL2 隔离执行。"""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        cfg = cfg or {}
        self.cfg = cfg
        ex = cfg.get("execution", {}) or {}
        self.enabled = bool(ex.get("enabled", False))
        self.max_expr_chars = int(ex.get("max_expr_chars", MAX_EXPR_CHARS))
        self.max_nodes = int(ex.get("max_nodes", MAX_NODES))
        self.max_output_chars = int(ex.get("max_output_chars", MAX_OUTPUT_CHARS))
        self.allow_latex = bool(ex.get("allow_latex", True))
        self.wsl2_enabled = bool(ex.get("wsl2", {}).get("enabled", False))

    def _check_enabled(self) -> None:
        if not self.enabled:
            raise ExecutionError(
                "执行网关默认关闭。轻量计算需求请先用检索/引用链路解决；"
                "确需开启时在 configs/experiment.yaml 设 execution.enabled=true。"
            )

    def calc(self, expr: str, variables: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self._check_enabled()
        t0 = time.time()
        try:
            value = safe_eval(expr, variables, self.max_expr_chars, self.max_nodes)
            out = str(value)
            if len(out) > self.max_output_chars:
                out = out[: self.max_output_chars] + "…(truncated)"
            return {"ok": True, "kind": "calc", "expr": expr, "output": out,
                    "error": "", "elapsed_ms": int((time.time() - t0) * 1000),
                    "expr_sha256": sha256_text(expr)[:16]}
        except ExecutionError as exc:
            return {"ok": False, "kind": "calc", "expr": expr, "output": "",
                    "error": str(exc), "elapsed_ms": int((time.time() - t0) * 1000),
                    "expr_sha256": sha256_text(expr)[:16]}
        except ZeroDivisionError as exc:
            return {"ok": False, "kind": "calc", "expr": expr, "output": "",
                    "error": f"除零: {exc}", "elapsed_ms": int((time.time() - t0) * 1000),
                    "expr_sha256": sha256_text(expr)[:16]}

    def latex(self, template: str, values: Dict[str, Any]) -> Dict[str, Any]:
        self._check_enabled()
        if not self.allow_latex:
            raise ExecutionError("LaTeX 渲染已在配置中关闭")
        t0 = time.time()
        try:
            out = render_latex(template, values, self.max_output_chars)
            return {"ok": True, "kind": "latex", "expr": template[:200], "output": out,
                    "error": "", "elapsed_ms": int((time.time() - t0) * 1000),
                    "expr_sha256": sha256_text(template)[:16]}
        except ExecutionError as exc:
            return {"ok": False, "kind": "latex", "expr": template[:200], "output": "",
                    "error": str(exc), "elapsed_ms": int((time.time() - t0) * 1000),
                    "expr_sha256": sha256_text(template)[:16]}

    def run(self, kind: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if kind in {"latex_compile", "plot"}:
            if not (self.cfg.get("execution", {}).get("docker_worker") or {}).get("enabled"):
                raise WSL2Unavailable("真实编译/图表需要显式启用已核验的 Docker worker；没有本机执行降级。")
            self._check_enabled()
            if kind=="latex_compile" and not self.allow_latex:
                raise ExecutionError("LaTeX 已在配置中关闭")
            from .docker_worker import execute, WorkerUnavailable
            try:
                return execute(kind, payload, self.cfg)
            except WorkerUnavailable as exc:
                raise ExecutionError(str(exc)) from exc
        if kind == "python_static":
            return self.analyze_python(str(payload.get("source", "")))
        if kind == "calc":
            return self.calc(str(payload.get("expr", "")), dict(payload.get("variables") or {}))
        if kind == "latex":
            return self.latex(str(payload.get("template", "")), dict(payload.get("values") or {}))
        if kind == "python_isolated":
            raise WSL2Unavailable(
                "任意 Python 执行需要 WSL2 隔离（Spec §3.2 高风险路径）。"
                "当前未配置受审查的隔离 worker；安装 WSL2 不等于已形成安全边界。"
                "可显式请求 python_static 分析；不允许降级到本进程执行。"
            )
        raise ExecutionError(f"未知的执行类型: {kind}")

    def analyze_python(self, source: str) -> Dict[str, Any]:
        """Syntax/requirement inventory only; never imports or executes source."""
        self._check_enabled()
        if not isinstance(source, str) or len(source) > 20000:
            raise ExecutionError("静态分析源码超过预算")
        try:
            tree = ast.parse(source, mode="exec")
        except (SyntaxError, RecursionError, MemoryError) as exc:
            return {"ok": False, "kind": "python_static", "executed": False,
                    "source_sha256": sha256_text(source), "error": type(exc).__name__}
        nodes = list(ast.walk(tree))
        if len(nodes) > 2000:
            raise ExecutionError("静态分析 AST 超过预算")
        imports, calls = set(), set()
        for node in nodes:
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.add(node.module or "relative-import")
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    calls.add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    calls.add(node.func.attr)
        return {"ok": True, "kind": "python_static", "executed": False,
                "source_sha256": sha256_text(source), "imports": sorted(imports),
                "calls": sorted(calls), "node_count": len(nodes),
                "security_approval": False, "isolation_worker_configured": False}


def build_gateway(cfg: Dict[str, Any]) -> ExecutionGateway:
    return ExecutionGateway(cfg)
