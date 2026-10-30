"""模型 provider 抽象与 deterministic fake provider。

M1 原则（Spec 附录 D 第 3 步）：**零成本、不联网、可重放**。
CI 与 E2E 一律走 FakeProvider；真实 provider 未接入时显式报错，不静默降级。

FakeProvider 不是"假装在进化"，而是一个**可控的行为模拟器**：
系统提示词里出现哪些指令关键词，会确定性地切换答题策略
（表层相似度匹配 vs. 逐项证据比对 vs. 是否引用），
从而让"改 System Prompt → 指标变化"这条因果链可被离线验证。
真实 GEPA/LLM 反思接入后，本类只用于测试与回归。
"""

from __future__ import annotations

import hashlib
import copy
import json
import os
import pathlib
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence
from urllib.parse import urlparse

from ..eval.options import explicit_information_gap_index
from ..util import artifacts_path, ensure_dir, sha256_obj, write_json
from .call_ledger import (end_http_attempt, record_operation_end,
                          start_http_attempt,
                          start_operation)
from .call_ledger import (end_http_attempt, record_operation_end,
                          start_http_attempt, start_operation)

STOPWORDS = {
    "the", "and", "for", "that", "with", "this", "from", "have", "has", "been",
    "are", "was", "were", "which", "what", "when", "into", "using", "used",
    "may", "can", "should", "following", "these", "those", "than", "then",
    "you", "your", "after", "before", "over", "under", "about", "their",
    "not", "but", "all", "any", "how", "why", "who", "its", "it's", "of",
    "in", "on", "to", "as", "at", "by", "or", "is", "be", "if", "a", "an",
}

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-]{2,}")

# 名义单价：只用于记账，**不是真实账单**（P2）。
#
# 对齐开源做法：gepa / paper-qa / dspy 都**不自己维护价目表** —— gepa 直接调
# `litellm.completion_cost()`（`src/gepa/lm.py:133`），paper-qa 用真 tokenizer
# （tiktoken `cl100k_base`，`src/paperqa/types.py:20`）。本项目当前是离线确定性
# 模型，没有真实账单可言，因此按 gepa 对 "callable LM" 的处理方式：
# **算不出来就承认算不出来**，给出 `usd=None` + `cost_estimated=True`，
# 而不是硬凑一个看似精确的数字。接真实 API 后改为 `litellm.completion_cost`。
PRICE_PER_1K = {"prompt": 0.0015, "completion": 0.002}

COST_BASIS = (
    "名义单价估算（非真实账单）：prompt $0.0015/1k、completion $0.002/1k，"
    "token 数由本地 len(text)/4 近似得出，未使用真实 tokenizer"
)

# 字面陈述判定的最小长度：比这更短的选项做子串匹配不可靠（会把 "Yes" 之类到处命中），
# 因此低于此长度时不参与"是否被陈述"的判定，也不据此弃权。
STATED_MIN_CHARS = 8

# 【N3】评分/弃权判据版本 —— 进入 final 的"协议指纹"。改判据必须递增。
# v1 = `raw_scores[best] == 0` 才算弃权（会把"分数并列"误判成无证据）；
# v2 = 所选选项**未被任何段落字面陈述**才算弃权（M4+ #3 修复）。
SCORING_RULE_VERSION = "stated-abstain-v2"

# 【N4】证据呈现规则版本 —— 也进"协议指纹"。截断方式决定模型看到了什么，
# 与评分判据同等重要，改了就必须递增，否则会制造静默协议漂移。
# v1 = 硬截断 `t[:6000] + " …[truncated]"`，从字符中间切且不说明丢了多少；
# v2 = 段落/句子边界截断 + 显式声明"后 N 字符已省略"；阈值从代码默认值
#      提升为 configs 显式项（见 models.task.max_context_chars）。
EVIDENCE_RULE_VERSION = "boundary-truncate-v2"


class ProviderError(RuntimeError):
    pass


def answer_json_schema_response_format(
    options: Sequence[str],
    valid_doc_ids: Sequence[str] = (),
    *,
    citation_enabled: bool = False,
    abstain_enabled: bool = False,
) -> Dict[str, Any]:
    """Build a strict, per-task response schema from program-owned inputs."""
    if not options:
        raise ProviderError("structured answer output requires at least one option")
    answer_indices = list(range(len(options)))
    if abstain_enabled:
        answer_indices.append(-1)
    properties: Dict[str, Any] = {
        "answer_index": {"type": "integer", "enum": answer_indices},
    }
    required = ["answer_index"]
    if citation_enabled:
        properties["cited_doc_ids"] = {
            "type": "array",
            "maxItems": 3,
            "items": ({
                "type": "string",
                "enum": list(dict.fromkeys(str(value) for value in valid_doc_ids if str(value))),
            } if valid_doc_ids else {"type": "string"}),
        }
        required.append("cited_doc_ids")
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "rsi_task_answer",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


def parse_structured_answer(
    content: str,
    options: Sequence[str],
    valid_doc_ids: Sequence[str] = (),
    *,
    citation_enabled: bool = False,
    abstain_enabled: bool = False,
) -> tuple[int, List[str], bool]:
    """Validate structured output without guessing or coercing model values."""
    def reject_duplicate_keys(pairs: List[tuple[str, Any]]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON field: {key}")
            result[key] = item
        return result

    try:
        value = json.loads(content, object_pairs_hook=reject_duplicate_keys)
    except (json.JSONDecodeError, TypeError, ValueError):
        return -1, [], True
    expected_keys = {"answer_index"}
    if citation_enabled:
        expected_keys.add("cited_doc_ids")
    if not isinstance(value, dict) or set(value) != expected_keys:
        return -1, [], True
    index = value.get("answer_index")
    if type(index) is not int:
        return -1, [], True
    if index == -1 and abstain_enabled:
        pass
    elif not 0 <= index < len(options):
        return -1, [], True
    cited: List[str] = []
    if citation_enabled:
        raw_citations = value.get("cited_doc_ids")
        allowed = {str(value) for value in valid_doc_ids}
        if (not isinstance(raw_citations, list) or len(raw_citations) > 3
                or any(not isinstance(item, str) or item not in allowed
                       for item in raw_citations)
                or len(raw_citations) != len(set(raw_citations))):
            return -1, [], True
        cited = list(raw_citations)
    return index, cited, False


def tokens(text: str) -> set:
    return {t for t in _TOKEN_RE.findall((text or "").lower()) if t not in STOPWORDS}


def overlap(a: set, b: set) -> int:
    return len(a & b)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().rstrip(".").lower())


def _stated_options(options: Sequence[str], docs: Sequence[Dict[str, Any]]) -> set:
    """哪些选项被检索到的段落**字面陈述**（归一化后的子串匹配）。"""
    hay = " ".join(_norm(d.get("text", "")) for d in docs)
    out = set()
    for i, opt in enumerate(options):
        needle = _norm(opt)
        if len(needle) >= 8 and needle in hay:
            out.add(i)
    return out


@dataclass
class ModelOutput:
    answer_index: int
    answer_text: str
    cited_doc_ids: List[str] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    mode: str = ""
    raw: str = ""
    # 真实 provider 自带的计费信息（P2）。None ⇒ 走 `cost_of()` 的名义估算路径。
    # 由 provider 自行填写，字段与 `cost_of()` 返回结构一致。
    cost_override: Dict[str, Any] | None = None
    pi_session_path: str = ""
    pi_operation_id: str = ""
    pi_trace: Dict[str, Any] = field(default_factory=dict)
    provider_http_attempts: int = 0
    parse_failed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "answer_index": self.answer_index,
            "answer_text": self.answer_text,
            "cited_doc_ids": list(self.cited_doc_ids),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "latency_ms": self.latency_ms,
            "mode": self.mode,
            "parse_failed": bool(self.parse_failed),
        }
        if self.cost_override:
            d["cost"] = dict(self.cost_override)
        if self.pi_session_path:
            d["pi_session_path"] = self.pi_session_path
            d["pi_operation_id"] = self.pi_operation_id
            d["pi_trace"] = dict(self.pi_trace)
        if self.provider_http_attempts:
            d["provider_http_attempts"] = int(self.provider_http_attempts)
        return d


@dataclass
class ChatOutput:
    """Measured free-form model call used by runtime routing and retrieval."""

    content: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int
    cost: Dict[str, Any]
    cache_hit: bool = False
    usage_available: bool = True
    http_attempts: int = 0
    # Exact model identifier echoed by the endpoint when available. Calibration
    # and annotation runners use this to fail closed on provider-side routing.
    model_id: str = ""
    # Provider completion status, e.g. OpenAI-compatible `stop` or `length`.
    # Optional for compatibility with injected/test providers and old endpoints.
    finish_reason: Optional[str] = None


class FakeProvider:
    """可控的行为模拟器。

    两种模式，务必分清（别在报告里说反）：
    1. **noise=0（默认，M1 行为）**：完全确定性，seed 不影响结果。
       用于链路验证与逐字节重放。
    2. **noise>0（M2 多 seed 用）**：按 `hash(seed|task_id)` 稳定地让一小部分题目
       "抖动"到次优选项，模拟真实模型解码/采样带来的题级随机性。

    注意：这个噪声**不是真实模型方差**，只是让"多 seed + 配对 bootstrap CI"
    这条统计链路在离线环境下可被端到端验证。接真实 provider 后，
    `noise` 必须回到 0，方差由真实采样产生。
    """

    def __init__(self, model_id: str = "fake-deterministic-v1", seed: int = 0, noise: float = 0.0,
                 channels: Any = None):
        self.model_id = model_id
        self.seed = int(seed)
        self.noise = float(noise)
        from .channels import load_channels

        self.channels = channels or load_channels(None)

    def _jitter(self, task_id: str) -> bool:
        """题级抖动：同一 (seed, task_id) 永远同结果，不同 seed 相互独立。"""
        if self.noise <= 0.0:
            return False
        h = int(hashlib.sha256(f"{self.seed}|{task_id}|jitter".encode()).hexdigest()[:8], 16)
        return (h / 0xFFFFFFFF) < self.noise

    # 指令开关：v0 基线提示词刻意不含这些关键词
    #
    # 【M3】行为开关有两个来源：System Prompt 的措辞 ∪ 命中 Skill 正文里的程序步骤。
    # 【M3.1】并集语义被**通道划分**取代：每个行为有唯一归属通道，
    # 只从归属通道读取（见 runtime/channels.py）。原因见那里的说明 ——
    # 并集会让 prompt / body / 新 Skill 三个维度落进同一批开关，
    # 消融因此检不出交互效应。legacy 并集口径用 `behavior_channels.enabled=false` 复现。
    def _modes(self, system_prompt: str, skill_body: str = "",
               channels: Any = None) -> Dict[str, bool]:
        from .channels import effective_modes

        return effective_modes(system_prompt, skill_body, channels or self.channels)

    def complete(self, *, system_prompt: str, question: str, options: Sequence[str],
                 docs: Sequence[Dict[str, Any]], context: str = "", task_id: str = "",
                 skill_body: str = "", channels: Any = None) -> ModelOutput:
        modes = self._modes(system_prompt, skill_body, channels)
        q_tokens = tokens(question)
        doc_tokens = tokens(" ".join(d.get("text", "") for d in docs))

        if modes["evidence"]:
            # 逐项证据比对：把 top_k 里的**全部**段落都算进来
            basis = doc_tokens
            mode = "evidence_scan"
        else:
            # 表层匹配：不系统比对，但上下文照样在 prompt 里 —— 模型会读最靠前那一篇。
            # 这个区别是 M2 夹具的关键：Skill 召回决定"最靠前那篇是不是 gold"，
            # 因此"改 description → 召回率变化 → 正确率变化"这条因果链才真实存在。
            top1 = docs[0].get("text", "") if docs else ""
            basis = q_tokens | tokens(top1)
            mode = "surface_match"

        # 逐项证据比对的**实质判据**：选项是否被检索到的段落直接陈述。
        #
        # 只用"词集合重叠"来模拟比对是不够的 —— 干扰项只要在段落里出现几个词就能拿分，
        # 于是"比不比对"结果一样，A1/A3 消融会恒等于 0（M2 实测正是如此）。
        # 真实证据比对看的是"这段话是不是直接给出了这个结论"，这里用字面陈述来近似：
        # gold 文档里通常有 "Reported finding: <选项>" 这样的完整陈述，
        # 干扰项文档只有改写/部分提及。这条判据让"证据进没进 top_k"真正决定对错。
        #
        # 【M4+】不再只在 evidence 模式下计算：弃权判据同样依赖它（见下面的 abstain）。
        stated = _stated_options(options, docs)

        scores: List[int] = []
        raw_scores: List[int] = []
        for i, opt in enumerate(options):
            s = overlap(tokens(opt), basis)
            # 【M1/M2 回顾】这个 +10000 必须**只在 evidence 行为生效时**加。
            # 它在 M4+ 修弃权判据时被改成无条件生效，后果是逐项比对行为
            # （skill_body 通道的 `evidence`）被抹平：开与不开答案选同一个，
            # 实测 18/18 题 net=0 → skill.misapplied 分支永远提不出候选。
            # 而弃权判据只需要 `stated` 这个集合本身，不需要它参与打分。
            if modes["evidence"] and i in stated:
                s += 10000
            raw_scores.append(overlap(tokens(opt), basis))
            # 稳定微扰，避免完全平局导致退化（同一 task 永远同一结果）
            s = s * 1000 + (int(hashlib.sha256(f"{task_id}|{opt}".encode()).hexdigest()[:4], 16) % 7)
            scores.append(s)

        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        best = order[0]
        if self._jitter(task_id) and len(order) > 1:
            best = order[1]      # 抖动：退到次优选项（模拟采样噪声）
        chosen = options[best]

        cited: List[str] = []
        if modes["citation"]:
            opt_tokens = tokens(chosen)
            for d in docs:
                if overlap(opt_tokens, tokens(d.get("text", ""))) > 0:
                    cited.append(d.get("doc_id", ""))
                if len(cited) >= 2:
                    break

        prompt_tokens = len(tokens(system_prompt)) + len(q_tokens) + sum(len(tokens(o)) for o in options) \
            + sum(len(tokens(d.get("text", ""))) for d in docs) // 4
        completion_tokens = max(8, len(tokens(chosen)))

        # 弃权（abstain）：**我正要输出的这个选项，在检索到的任何段落里都没有被直接陈述**。
        #
        # 【M4+】判据重写。旧判据是「最佳选项与检索段落零实词重叠」—— 太严，实测
        # **从来不触发**：干扰项（如 "Related work on <topic>"）几乎总能蹭到一个实词，
        # 于是终评上 abstain_rate 0.0→0.0、unsupported_answer_rate 1.0→1.0，
        # "没有证据就别猜"这条机制在数字上是死的，进化只证明了"改召回能涨分"。
        #
        # 新判据对齐 paper-qa（context 不足就回 CANNOT_ANSWER_PHRASE）与
        # asta-bench LitQA2（把"信息不足"做成显式选项，用 coverage × precision 双指标约束）：
        # 判的是"我要给的答案有没有被证据直接说出来"，而不是"有没有词撞上"。
        #
        # 两个保护：
        #   1. 选项太短（< STATED_MIN_CHARS）时字面陈述判定不可靠，不据此弃权；
        #   2. 弃权只在"该选项确实可判"时才发生，避免把本来答对的题改成弃权。
        if modes["abstain"] and len(_norm(chosen)) >= STATED_MIN_CHARS and best not in stated:
            return ModelOutput(
                answer_index=-1, answer_text="", cited_doc_ids=[], prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens, latency_ms=120, mode=mode + "+abstain",
                raw="INSUFFICIENT_EVIDENCE",
            )

        return ModelOutput(
            answer_index=best,
            answer_text=chosen,
            cited_doc_ids=[c for c in cited if c],
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=90 + len(docs) * 15,
            mode=mode,
            raw=chosen,
        )


# ---------------------------------------------------------------------------
# 真实 HTTP provider（OpenAI 兼容端点）
# ---------------------------------------------------------------------------
#
# 设计取舍，逐条写清楚，免得后人误读：
#
# 1. **不新增第三方依赖**：用 `requests` 直连 `/v1/chat/completions`。
#    对齐 gepa —— 它也不自己写 HTTP，而是把各家 SDK 统一在 `litellm` 后面
#    （`src/gepa/lm.py`），本项目只是把"统一层"换成 20 行 requests，
#    等需要多厂商时再换 litellm，属于可替换实现，不是设计缺陷。
#
# 2. **关闭 thinking 必须用顶层 `enable_thinking: false`**。
#    硅基流动上 `chat_template_kwargs` 被静默忽略（实测 reasoning 仍 372 token），
#    `thinking_budget: 0` 直接 400。这个坑不写下来，换个人跑就会以为"关不掉"。
#
# 3. **通道语义必须与 FakeProvider 一致**：要不要引用、要不要弃权，
#    由 `effective_modes()` 从 system_prompt / skill_body 按通道判定后决定
#    是否把对应指令拼进 prompt —— 而不是无脑全拼。
#    否则 `abstain` 通道的消融在真实模型上会失效（所有候选都弃权）。
#
# 4. **响应落盘缓存**：真实调用不可重放，缓存是唯一能让"重跑同协议"不花钱、
#    且结果逐字节一致的手段。键 = 请求全量哈希。
#
# 5. **解析失败绝不静默猜一个选项**：计入 `+parse_fail`，交由调用方决定。
#
DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_API_KEY_ENV = "SILICONFLOW_API_KEY"

# ---------------------------------------------------------------------------
# 密钥注入：`.env`（项目根，不入库）
#
# 为什么需要这条通路：密钥**既不写进仓库配置**，也不该出现在命令行或对话记录里。
# 但自动化执行时，子进程未必继承到交互式终端里 export 的变量 —— 于是真实跑只能
# 由人手工执行。这里让代码自己从项目根 `.env` 读一次，把"人手工跑"变成"配一次、
# 之后全自动"。
#
# 安全约束（逐条都是硬要求）：
#   1. 只做 `setdefault`：进程里已存在的同名变量**优先**，.env 不覆盖（显式优先）。
#   2. 文件不存在 / 解析失败 → 静默跳过，绝不抛错、绝不打印任何值。
#   3. `.env` 必须已在 .gitignore 中（见仓库根 .gitignore）。
#   4. 任何日志、报告、异常信息里都不出现密钥内容。
# ---------------------------------------------------------------------------
_DOTENV_LOADED = False


def _project_root() -> str:
    return str(pathlib.Path(__file__).resolve().parents[2])


def load_project_dotenv(filename: str = ".env") -> int:
    """从项目根 `.env` 补齐缺失的环境变量。返回实际注入的键数。

    幂等；已注入过的键不再重复解析。解析规则宽松：忽略空行与 `#` 注释，
    允许行首 `export `，值两侧的单/双引号会被剥掉。
    """
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return 0
    _DOTENV_LOADED = True
    try:
        p = os.path.join(_project_root(), filename)
        if not os.path.exists(p):
            return 0
        n = 0
        with open(p, "r", encoding="utf-8") as fh:
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[len("export "):].strip()
                if "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip("'\"").strip()
                if not key or key in os.environ:
                    continue          # 显式环境变量优先；.env 不覆盖
                os.environ[key] = val
                n += 1
        return n
    except Exception:
        return 0                      # 绝不因 .env 问题让主流程崩


# 模块导入即注入一次，确保 HttpProvider 构造时密钥已就位
load_project_dotenv()

# SiliconFlow published serverless list tariffs (USD / 1M tokens), checked
# 2026-10-05 on its official per-model pages. Apply only to the international
# SiliconFlow API host; model name and vendor alone do not determine price. Existing
# Qwen3 tariffs were rechecked on 2026-10-15 and Qwen3.5-122B was added from
# its current official model page on the same date.
# These are tariff-priced amounts from API-reported usage, not account invoices.
SILICONFLOW_PRICE_SOURCE = "https://www.siliconflow.com/models/{slug}"
SILICONFLOW_BILLING_SOURCE = (
    "https://docs.siliconflow.cn/docs/userguide/capabilities/text-generation")
SILICONFLOW_PRICE_CHECKED_AT = "2026-10-15"
SILICONFLOW_API_HOSTS = frozenset({"api.siliconflow.cn", "api.siliconflow.com"})

# DeepSeek's direct API publishes peak and off-peak rates.  The runtime only
# receives aggregate input/output usage, so the aggregate prompt rate uses the
# highest (peak cache-miss) input rate as a conservative upper bound.  The
# cache_read field still records the distinct official peak cache-hit rate for
# callers that have a cache breakdown; this provider currently has no such
# breakdown in its usage payload.
DEEPSEEK_PRICE_SOURCE = "https://api-docs.deepseek.com/quick_start/pricing"
DEEPSEEK_BILLING_SOURCE = DEEPSEEK_PRICE_SOURCE
DEEPSEEK_PRICE_CHECKED_AT = "2026-10-18"
DEEPSEEK_API_HOST = "api.deepseek.com"
DEEPSEEK_FLASH_PEAK_INPUT_CACHE_HIT_USD_PER_1M = 0.006
DEEPSEEK_FLASH_PEAK_INPUT_CACHE_MISS_USD_PER_1M = 0.3
DEEPSEEK_FLASH_PEAK_OUTPUT_USD_PER_1M = 1.2
DEEPSEEK_PRICE_PER_1M: Dict[str, Dict[str, float]] = {
    "deepseek-flash": {
        "prompt": DEEPSEEK_FLASH_PEAK_INPUT_CACHE_MISS_USD_PER_1M,
        "completion": DEEPSEEK_FLASH_PEAK_OUTPUT_USD_PER_1M,
        "cache_read": DEEPSEEK_FLASH_PEAK_INPUT_CACHE_HIT_USD_PER_1M,
        "cache_write": DEEPSEEK_FLASH_PEAK_INPUT_CACHE_MISS_USD_PER_1M,
    },
}


def _endpoint_post(requests_module: Any, url: str, **kwargs: Any) -> Any:
    """Bypass ambient proxies only for the official SiliconFlow API hosts.

    Windows proxy environment settings can route TLS through a local proxy that
    fails the direct SiliconFlow handshake. Other configured providers retain
    Requests' default environment-proxy behavior.
    """
    host = (urlparse(str(url)).hostname or "").lower()
    if host not in SILICONFLOW_API_HOSTS:
        return requests_module.post(url, **kwargs)
    with requests_module.Session() as session:
        session.trust_env = False
        return session.post(url, **kwargs)


# CN and international accounts have different currencies and tariffs. These
# three zero-price CN rows were independently read from the official CN pricing
# page, not inferred from model size or international prices. Zero is currency
# invariant; nonzero CNY prices cannot be returned as USD without an explicit FX
# protocol. Evidence: artifacts/provenance/siliconflow-cn-pricing-20261015.json.
SILICONFLOW_CN_FREE_MODELS = frozenset({
    "Qwen/Qwen3-8B", "Qwen/Qwen2.5-7B-Instruct", "THUDM/GLM-4-9B-0414",
    "XingChenAGI/Xing4.0-29B",
})
SILICONFLOW_CN_PRICE_SOURCE = "https://siliconflow.cn/pricing"
SILICONFLOW_CN_PRICE_EVIDENCE = "5006960476668bdea624d89604e17709fd657f170b8270d67aaa9028afe9b74e"
SILICONFLOW_CN_PRICE_EVIDENCE_BY_MODEL = {
    "Qwen/Qwen3-8B": SILICONFLOW_CN_PRICE_EVIDENCE,
    "Qwen/Qwen2.5-7B-Instruct": SILICONFLOW_CN_PRICE_EVIDENCE,
    "THUDM/GLM-4-9B-0414": SILICONFLOW_CN_PRICE_EVIDENCE,
    "XingChenAGI/Xing4.0-29B": "3d5cdfecf307f5ea088b0c93ed2f7607ca31367dc2d5ffd152b015b4b9c0aba7",
}
PRICE_PER_1M: Dict[str, Dict[str, float]] = {
    # These model pages publish a single input and output rate, with no separate
    # cache rate. SiliconFlow's billing formula prices input tokens at that
    # input rate, so cacheRead/cacheWrite tokens use the same published rate.
    "Qwen/Qwen3-8B": {"prompt": 0.06, "completion": 0.06,
                      "cache_read": 0.06, "cache_write": 0.06},
    "Qwen/Qwen3-32B": {"prompt": 0.14, "completion": 0.57,
                       "cache_read": 0.14, "cache_write": 0.14},
    "Qwen/Qwen3.5-122B-A10B": {"prompt": 0.26, "completion": 2.08,
                                "cache_read": 0.26, "cache_write": 0.26},
    "deepseek-ai/DeepSeek-V3.2": {"prompt": 0.27, "completion": 0.42,
                                   "cache_read": 0.135, "cache_write": 0.27},
    "zai-org/GLM-5.3-Flash": {"prompt": 0.15, "completion": 0.5,
                               "cache_read": 0.03, "cache_write": 0.15},
}


def published_price_for_endpoint(model: str, base_url: str) -> Dict[str, float] | None:
    """Return applicable USD rates; never silently convert a nonzero CN tariff."""
    try:
        hostname = (urlparse(str(base_url)).hostname or "").lower()
    except ValueError:
        return None
    if hostname == DEEPSEEK_API_HOST:
        price = DEEPSEEK_PRICE_PER_1M.get(str(model))
        return dict(price) if price is not None else None
    if hostname not in SILICONFLOW_API_HOSTS:
        return None
    if hostname == "api.siliconflow.cn":
        return ({key: 0.0 for key in ("prompt", "completion", "cache_read", "cache_write")}
                if str(model) in SILICONFLOW_CN_FREE_MODELS else None)
    price = PRICE_PER_1M.get(str(model))
    return dict(price) if price is not None else None


def published_tariff_metadata(model: str, base_url: str) -> Dict[str, Any]:
    """Identify the tariff source shared by HTTP and Pi accounting."""
    if published_price_for_endpoint(model, base_url) is None:
        return {}
    hostname = (urlparse(str(base_url)).hostname or "").lower()
    if hostname == DEEPSEEK_API_HOST:
        return {
            "pricing_provider": "DeepSeek",
            "pricing_source": DEEPSEEK_PRICE_SOURCE,
            "pricing_formula_source": DEEPSEEK_BILLING_SOURCE,
            "pricing_checked_at": DEEPSEEK_PRICE_CHECKED_AT,
            "pricing_is_account_invoice": False,
            "pricing_region": "international",
            "pricing_source_currency": "USD",
            "pricing_currency_conversion": "none",
            "pricing_peak_upper_bound": True,
            "pricing_input_cache_hit_peak_usd_per_1m":
                DEEPSEEK_FLASH_PEAK_INPUT_CACHE_HIT_USD_PER_1M,
            "pricing_input_cache_miss_peak_usd_per_1m":
                DEEPSEEK_FLASH_PEAK_INPUT_CACHE_MISS_USD_PER_1M,
            "pricing_output_peak_usd_per_1m": DEEPSEEK_FLASH_PEAK_OUTPUT_USD_PER_1M,
        }
    cn = hostname == "api.siliconflow.cn"
    slugs = {"Qwen/Qwen3-8B": "qwen3-8b", "Qwen/Qwen3-32B": "qwen3-32b",
             "Qwen/Qwen3.5-122B-A10B": "qwen3-5-122b-a10b",
             "deepseek-ai/DeepSeek-V3.2": "deepseek-v3-2",
             "zai-org/GLM-5.3-Flash": "glm-5-3-flash"}
    return {"pricing_provider": "SiliconFlow",
            "pricing_source": (SILICONFLOW_CN_PRICE_SOURCE if cn else
                               SILICONFLOW_PRICE_SOURCE.format(slug=slugs[model])),
            "pricing_formula_source": SILICONFLOW_BILLING_SOURCE,
            "pricing_checked_at": SILICONFLOW_PRICE_CHECKED_AT,
            "pricing_is_account_invoice": False,
            "pricing_region": "CN" if cn else "international",
            "pricing_source_currency": "CNY" if cn else "USD",
            "pricing_currency_conversion": "zero_price_currency_invariant" if cn else "none",
            **({"pricing_evidence_report_sha256": SILICONFLOW_CN_PRICE_EVIDENCE_BY_MODEL[model]}
               if cn else {})}

_ABSTAIN_TOKENS = {"insufficient", "insufficient_information", "none", "unknown", "n/a", "abstain"}
_LETTER_OFFSET = ord("A")


def _boundary_index(t: str, limit: int) -> int:
    """在 `limit` 内找最后一个"干净"的截断点：优先段落、其次行、再次句末。

    找不到足够靠后的边界（早于 limit 的一半）就退回硬切 —— 宁可切得难看，
    也不能为对齐边界把证据砍掉一大半。
    """
    window = t[:limit]
    floor = int(limit * 0.5)
    for sep in ("\n\n", "\n"):
        i = window.rfind(sep)
        if i >= floor:
            return i + len(sep)
    best = -1
    for i, ch in enumerate(window):
        if ch in ".。!！?？;；":
            best = i
    return best + 1 if best >= floor else limit


def _truncate(text: str, limit: int) -> str:
    """按段落/句子边界截断，并**显式声明**证据不完整。

    旧实现 `t[:limit] + " …[truncated]"` 有两个缺陷：
    1. **从字符中间硬切** —— 实验步骤、结论句被劈成两半，模型拿到残缺语义；
    2. **标注含糊** —— 挂在尾部且不说明丢了多少，模型常常意识不到证据被截过，
       于是**在残缺证据上照答不误**。

    protocolqa 实测（dev 61 题消融，唯一变量 = 本阈值，走真实 harness）：
    被截的 18 题 acc 0.278→0.389、弃权率 0.556→0.222；未被截的 43 题**零变化**。
    该集证据 p50 5720 / max 14188 字符，旧阈值 6000 下 selection_gate 有 38%
    的题被截，最多丢掉 57.6% 的正文 —— 这不是"模型读不出长上下文"，是证据
    被物理切断。litqa2 / pubmedqa 一题未截，属 protocolqa 独有。
    """
    t = (text or "").strip()
    if len(t) <= limit:
        return t
    cut = _boundary_index(t, limit)
    omitted = len(t) - cut
    return (f"{t[:cut].rstrip()}\n"
            f"[… 后 {omitted} 字符已省略：本条证据不完整。"
            f"若关键步骤落在被省略的部分，请按信息不足处理]")


class HttpProvider:
    """OpenAI 兼容端点的真实 provider（默认硅基流动）。

    配置项全部来自 `configs/experiment.yaml: models.task` 下的键，
    密钥**只从环境变量读**（`api_key_env`），禁止写进仓库配置文件。
    """

    def __init__(self, model_id: str = "Qwen/Qwen3-8B", base_url: str = DEFAULT_BASE_URL,
                 api_key: str = "", api_key_env: str = DEFAULT_API_KEY_ENV,
                 temperature: float = 0.0, max_tokens: int = 512, timeout: float = 120.0,
                 max_retries: int = 6, enable_thinking: bool = False,
                 extra_body: Dict[str, Any] | None = None,
                 pricing_override: Dict[str, float] | None = None,
                 pricing_metadata: Dict[str, Any] | None = None,
                 structured_output: str = "",
                 max_context_chars: int = 6000, max_doc_chars: int = 1500,
                 cache: bool = True, cooldown: float = 0.0, channels: Any = None,
                 send_run_seed: bool = False,
                 run_seed: int = 0, **_: Any):
        self.model_id = model_id
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.api_key = api_key or os.environ.get(api_key_env, "")
        if not self.api_key:
            raise ProviderError(
                f"未找到 API key：请设置环境变量 {api_key_env}。"
                "密钥不得写入 configs/ 或任何产物文件。"
            )
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.timeout = float(timeout)
        self.max_retries = int(max_retries)
        self.enable_thinking = bool(enable_thinking)
        self.extra_body = dict(extra_body or {})
        # Optional caller-frozen tariff for an endpoint/model absent from the
        # built-in provider table. Secrets are never part of this metadata.
        self.pricing_override = (dict(pricing_override)
                                 if pricing_override is not None else None)
        self.pricing_metadata = dict(pricing_metadata or {})
        self.structured_output = str(structured_output or "").strip().lower()
        if self.structured_output not in {"", "json_schema"}:
            raise ProviderError("structured_output must be empty or json_schema")
        self.max_context_chars = int(max_context_chars)
        self.max_doc_chars = int(max_doc_chars)
        # Cache namespaces always include the run seed.  For providers that
        # implement OpenAI's optional ``seed`` field, callers may also opt in
        # to forwarding it on the wire; the default stays off for legacy
        # endpoints that reject unknown request fields.
        self.send_run_seed = bool(send_run_seed)
        self.run_seed = int(run_seed)
        from .channels import load_channels

        self.channels = channels or load_channels(None)
        self.cache = bool(cache) and os.environ.get("RSI_HTTP_NO_CACHE", "") != "1"
        # 可选：每次成功调用后的小冷却（秒），缓解免费档 RPM 窗口限流。
        # 默认 0；重跑大批量时可经 RSI_HTTP_COOLDOWN 环境变量设置。
        self.cooldown = float(cooldown) or float(
            os.environ.get("RSI_HTTP_COOLDOWN", "0") or 0)
        self.cache_dir = artifacts_path("http_cache")
        self.stats = {"calls": 0, "cache_hits": 0, "retries": 0, "parse_fail": 0, "errors": 0}
        self._last_http_attempts = 0
        self._last_usage_complete = True

    # -- 请求构造 -----------------------------------------------------------
    def _messages(self, *, system_prompt: str, question: str, options: Sequence[str],
                  docs: Sequence[Dict[str, Any]], context: str, skill_body: str,
                  channels: Any) -> List[Dict[str, str]]:
        from .channels import effective_modes

        modes = effective_modes(system_prompt, skill_body, channels or self.channels)
        sys_parts = [system_prompt.strip()]
        if skill_body.strip():
            # 【M3】渐进披露：正文只在召回后进入上下文
            sys_parts.append("Skill procedure (follow these steps):\n" + skill_body.strip())
        system = "\n\n".join(p for p in sys_parts if p)

        blocks: List[str] = []
        if context.strip():
            blocks.append("Context:\n" + _truncate(context, self.max_context_chars))
        if docs:
            label = ("Task-provided evidence" if all(
                d.get("source_kind") == "task_input" for d in docs)
                else "Retrieved passages")
            lines = [label + " (identifier in brackets):"]
            for d in docs:
                lines.append(f"[{d.get('doc_id','')}] {_truncate(d.get('text',''), self.max_doc_chars)}")
            blocks.append("\n".join(lines))
        else:
            blocks.append("Evidence passages: (none)")
        opts = "\n".join(f"{i}. {o}" for i, o in enumerate(options))
        blocks.append("Question: " + question.strip())
        blocks.append("Options:\n" + opts)

        # 【通道一致性】输出契约也按通道裁剪，避免"所有候选都弃权/都引用"
        if self.structured_output == "json_schema":
            fields = ['"answer_index": <integer option index>']
            if modes.get("citation"):
                fields.append('"cited_doc_ids": [<supporting passage identifiers>]')
            rules = ["Return one JSON object with exactly these fields and no prose:",
                     "{" + ", ".join(fields) + "}"]
            if modes.get("abstain") and explicit_information_gap_index(list(options)) is None:
                rules.append("Use answer_index -1 only when the supplied evidence cannot support an option.")
        else:
            line_count = "two lines" if modes.get("citation") else "one line"
            rules = [f"Reply with EXACTLY the following {line_count} and nothing else:",
                     "ANSWER: <option number>"]
            if modes.get("citation"):
                rules.append("CITED: <comma-separated passage identifiers that support your answer>")
        if modes.get("abstain") and self.structured_output != "json_schema":
            gap_index = explicit_information_gap_index(list(options))
            if gap_index is not None:
                gap_label = str(options[gap_index]).strip()
                rules.append(
                    "If the supplied evidence does not establish whether the claim "
                    "is supported or contradicted, select the explicit "
                    f"information-gap class {gap_label} at option {gap_index} "
                    "using its option number; do not use runtime abstention.")
            else:
                rules.append("If no supplied passage directly states any option, "
                             "reply `ANSWER: INSUFFICIENT` instead of guessing.")
        blocks.append("Output format:\n" + "\n".join(rules))
        return [{"role": "system", "content": system},
                {"role": "user", "content": "\n\n".join(blocks)}]

    def chat(self, system: str, user: str, max_tokens: Optional[int] = None) -> str:
        """自由格式对话（反思 / 元任务专用）：**不走答题模板**。

        为什么不能复用 `complete()`：`_messages()` 会强制追加答题输出契约
        （"Reply with EXACTLY … ANSWER: <option number>"）。反思任务被套进去之后，
        模型会直接**答题**（实测原样输出 `ANSWER: 3`）而不是产出指令 ——
        它看到的最后一句要求是"只回两行"，我们真正的要求被挤到前面去了。

        本方法不读缓存：反思要的是多样性，命中缓存会让每轮拿到同一句。
        """
        body = self._body([{"role": "system", "content": system},
                           {"role": "user", "content": user}])
        if max_tokens:
            body["max_tokens"] = int(max_tokens)
        payload = self._post(body)
        try:
            return str(payload["choices"][0]["message"].get("content", ""))
        except Exception as exc:
            raise ProviderError(f"响应缺少 choices[0].message.content：{exc}")

    def chat_measured(self, system: str, user: str,
                      max_tokens: Optional[int] = None,
                      response_format: Optional[Dict[str, Any]] = None) -> ChatOutput:
        """Deterministic runtime meta-call with full usage and response caching."""
        body = self._body([{"role": "system", "content": system},
                           {"role": "user", "content": user}])
        if max_tokens is not None:
            body["max_tokens"] = int(max_tokens)
        if response_format is not None:
            # Per-call schemas must not mutate or leak into provider-wide
            # extra_body; the effective body also feeds the response cache key.
            effective_format = copy.deepcopy(response_format)
            # DeepSeek's current official chat endpoint advertises JSON output,
            # but rejects OpenAI's json_schema response_format (HTTP 400). Keep
            # the frozen schema in the plan and perform strict host validation;
            # only the wire-level request is reduced to the provider-supported
            # JSON object mode.
            if ((urlparse(self.base_url).hostname or "").lower() == "api.deepseek.com"
                    and effective_format.get("type") == "json_schema"):
                effective_format = {"type": "json_object"}
            body["response_format"] = effective_format
        key = self._cache_key(body)
        hit = self._cached(key)
        t0 = time.time()
        if hit is None:
            payload = self._post(body)
            http_attempts = int(self._last_http_attempts)
            latency_ms = int((time.time() - t0) * 1000)
            self.stats["calls"] += 1
            if self.cache:
                try:
                    ensure_dir(self.cache_dir)
                    write_json(self.cache_dir / f"{key}.json", payload)
                except Exception:
                    pass
        else:
            payload, latency_ms = hit, 0
            http_attempts = 0
            self.stats["cache_hits"] += 1
        try:
            choice = payload["choices"][0]
            content = str(choice["message"].get("content") or "")
            finish_reason_value = choice.get("finish_reason")
            finish_reason = (finish_reason_value if isinstance(finish_reason_value, str)
                             else None)
            usage = payload.get("usage") or {}
            usage_available = (
                usage.get("prompt_tokens") is not None
                and usage.get("completion_tokens") is not None)
            prompt_tokens = int(usage.get("prompt_tokens") or 0)
            completion_tokens = int(usage.get("completion_tokens") or 0)
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            self.stats["errors"] += 1
            raise ProviderError(f"meta-call response structure invalid: {exc}") from exc
        response_model = str(payload.get("model") or self.model_id)
        cost = self._cost_override(prompt_tokens, completion_tokens,
                                   response_model, self.base_url)
        cost["usage_available"] = usage_available
        if not usage_available:
            cost = {**cost, "usd": None, "usd_nominal": None,
                    "cost_estimated": True,
                    "cost_basis": "API usage counters unavailable; cost not estimable"}
        if hit is None and not getattr(self, "_last_usage_complete", True):
            cost = {**cost, "known_success_usd": cost.get("usd"),
                    "usd": None, "usd_nominal": None, "cost_estimated": True,
                    "usage_complete": False,
                    "cost_basis": "one or more dispatched attempts have unknown usage"}
        else:
            cost["usage_complete"] = usage_available
        if hit is not None:
            if usage_available:
                cost = {**cost, "usd": 0.0, "usd_nominal": 0.0,
                        "cost_estimated": False, "cost_basis": "response cache hit"}
        return ChatOutput(content, prompt_tokens, completion_tokens,
                          latency_ms, cost, cache_hit=hit is not None,
                          usage_available=usage_available,
                          http_attempts=http_attempts, model_id=response_model,
                          finish_reason=finish_reason)

    def _body(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        # 见文件头第 2 条：只有顶层 enable_thinking 真的生效
        body["enable_thinking"] = bool(self.enable_thinking)
        if self.send_run_seed:
            body["seed"] = self.run_seed
        body.update(self.extra_body)
        if (urlparse(self.base_url).hostname or "").lower() == "api.deepseek.com":
            # DeepSeek Flash defaults to thinking mode. The legacy
            # enable_thinking flag is ignored there; explicitly disable it for
            # bounded JSON rubric jobs unless the config opts in.
            body["thinking"] = {"type": "enabled" if self.enable_thinking else "disabled"}
        # ai_jobs freezes the requested schema in provider.extra_body for
        # task-first jobs; apply the same DeepSeek wire compatibility rule to
        # that path as to chat_measured(response_format=...).
        if ((urlparse(self.base_url).hostname or "").lower() == "api.deepseek.com"
                and isinstance(body.get("response_format"), dict)
                and body["response_format"].get("type") == "json_schema"):
            body["response_format"] = {"type": "json_object"}
        return body

    # -- 传输 ---------------------------------------------------------------
    def _cache_key(self, body: Dict[str, Any]) -> str:
        # Include every seed, including zero: a legacy cache entry has no
        # seed provenance and must not be reused as one M2 sample.
        identity = {"base_url": self.base_url, "body": body,
                    "run_seed": self.run_seed}
        return sha256_obj(identity)[:32]

    def _cached(self, key: str) -> Optional[Dict[str, Any]]:
        if not self.cache:
            return None
        f = self.cache_dir / f"{key}.json"
        if not f.exists():
            return None
        try:
            return json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _post(self, body: Dict[str, Any]) -> Dict[str, Any]:
        import requests  # 局部导入：离线跑 fake 时不强制依赖

        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}",
                   "Content-Type": "application/json"}
        last = ""
        self._last_http_attempts = 0
        # Usage uncertainty belongs to this request's attempts, not to every
        # later request made through the same provider instance.
        self._last_usage_complete = True
        operation_id = start_operation(None, "http", self.model_id)
        succeeded = False
        statuses: List[int] = []
        try:
            for attempt in range(self.max_retries):
                self._last_http_attempts = attempt + 1
                start_http_attempt(operation_id, attempt=attempt + 1)
                try:
                    r = _endpoint_post(
                        requests, url, headers=headers, json=body, timeout=self.timeout)
                except Exception as exc:                       # 网络抖动
                    self._last_usage_complete = False
                    end_http_attempt(operation_id, attempt=attempt + 1,
                                     status=None, error_type=type(exc).__name__)
                    last = f"transport error: {exc}"
                    delay = min(30.0, 1.5 * (attempt + 1))
                    try:
                        transport_base = max(0.0, float(os.environ.get(
                            "RSI_PROVIDER_TRANSPORT_BACKOFF_BASE_SECONDS", "0")))
                    except ValueError:
                        transport_base = 0.0
                    if transport_base:
                        delay = min(60.0, max(delay, transport_base * (2 ** attempt)))
                    if attempt + 1 < self.max_retries:
                        time.sleep(delay)
                    self.stats["retries"] += 1
                    continue
                end_http_attempt(operation_id, attempt=attempt + 1,
                                 status=int(r.status_code))
                statuses.append(int(r.status_code))
                if int(r.status_code) in {408, 500, 502, 503, 504}:
                    self._last_usage_complete = False
                if r.status_code == 200:
                    # 可选：每次成功调用后小冷却，避免免费档 RPM 窗口被突发打满
                    cd = float(self.cooldown if hasattr(self, "cooldown") else 0.0)
                    if cd > 0:
                        time.sleep(cd)
                    succeeded = True
                    return r.json()
                last = f"HTTP {r.status_code}: {r.text[:300]}"
                if r.status_code in (408, 409, 429, 500, 502, 503, 504):
                    if attempt + 1 >= self.max_retries:
                        break
                    self.stats["retries"] += 1
                    # 尊重服务端 Retry-After；否则指数退避（封顶 30s）
                    ra = r.headers.get("Retry-After")
                    if ra and str(ra).isdigit():
                        delay = min(60.0, float(ra))
                    else:
                        delay = min(30.0, 2.0 * (2 ** attempt))
                        if r.status_code == 429:
                            try:
                                busy_base = max(0.0, float(os.environ.get(
                                    "RSI_PROVIDER_429_BACKOFF_BASE_SECONDS", "0")))
                            except ValueError:
                                busy_base = 0.0
                            if busy_base:
                                delay = min(60.0, max(delay, busy_base * (2 ** attempt)))
                    time.sleep(delay)
                    continue
                error = ProviderError(f"provider 返回不可重试错误 {last}")
                error.provider_http_attempts = self._last_http_attempts
                error.provider_statuses = [int(r.status_code)]
                raise error
            error = ProviderError(
                f"provider 调用失败（最多尝试 {self.max_retries} 次）：{last}")
            error.provider_http_attempts = self._last_http_attempts
            error.provider_statuses = statuses
            raise error
        finally:
            record_operation_end(operation_id,
                                 status="success" if succeeded else "failed")

    # -- 解析 ---------------------------------------------------------------
    def _parse(self, content: str, options: Sequence[str],
               valid_doc_ids: Sequence[str], *, abstain_enabled: bool = False) -> tuple:
        """返回 (answer_index, cited_ids, parse_failed)。"""
        text = (content or "").strip()
        idx = None
        m = list(re.finditer(r"^\s*ANSWER\s*[:=]\s*([^\n\r]+)", text, flags=re.I | re.M))
        # A typed answer has one value. Picking the last of two ANSWER fields
        # silently turns a malformed response into a scored prediction.
        if len(m) != 1:
            return -1, [], True
        if m:
            tok = m[-1].group(1).strip().strip(".*`\"' ").lower()
            if tok in _ABSTAIN_TOKENS:
                return -1, [], not abstain_enabled
            if tok.lstrip("-").isdigit():
                n = int(tok)
                idx = n if 0 <= n < len(options) else None
            elif len(tok) == 1 and tok.isalpha():
                n = ord(tok.upper()) - _LETTER_OFFSET
                idx = n if 0 <= n < len(options) else None
        if idx is None:
            return -1, [], True

        cited: List[str] = []
        mc = list(re.finditer(r"CITED\s*[:=]\s*([^\n\r]+)", text, flags=re.I))
        if len(mc) > 1:
            return -1, [], True
        if mc:
            for tok in re.split(r"[,\s]+", mc[-1].group(1)):
                tok = tok.strip("[]()\"' `")
                if tok in valid_doc_ids and tok not in cited:
                    cited.append(tok)
        return idx, cited, False

    # -- 主入口 -------------------------------------------------------------
    def complete(self, *, system_prompt: str, question: str, options: Sequence[str],
                 docs: Sequence[Dict[str, Any]], context: str = "", task_id: str = "",
                 skill_body: str = "", channels: Any = None) -> ModelOutput:
        messages = self._messages(system_prompt=system_prompt, question=question,
                                  options=options, docs=docs, context=context,
                                  skill_body=skill_body, channels=channels)
        body = self._body(messages)
        from .channels import effective_modes

        modes = effective_modes(system_prompt, skill_body, channels or self.channels)
        valid_ids = [str(d.get("doc_id", "")) for d in docs if d.get("doc_id")]
        abstain_enabled = bool(modes.get("abstain")) and (
            explicit_information_gap_index(list(options)) is None)
        if self.structured_output == "json_schema":
            body["response_format"] = answer_json_schema_response_format(
                options, valid_ids, citation_enabled=bool(modes.get("citation")),
                abstain_enabled=abstain_enabled)
        key = self._cache_key(body)
        hit = self._cached(key)
        t0 = time.time()
        if hit is not None:
            payload, latency_ms = hit, 0
            http_attempts = 0
            self.stats["cache_hits"] += 1
        else:
            payload = self._post(body)
            http_attempts = int(self._last_http_attempts)
            latency_ms = int((time.time() - t0) * 1000)
            self.stats["calls"] += 1
            if self.cache:
                try:
                    ensure_dir(self.cache_dir)
                    write_json(self.cache_dir / f"{key}.json", payload)
                except Exception:
                    pass

        try:
            msg = payload["choices"][0]["message"]
            usage = payload.get("usage") or {}
        except Exception as exc:
            self.stats["errors"] += 1
            raise ProviderError(f"provider 响应结构异常: {exc}; raw={str(payload)[:300]}")

        content = msg.get("content") or ""
        if self.structured_output == "json_schema":
            idx, cited, failed = parse_structured_answer(
                content, options, valid_ids,
                citation_enabled=bool(modes.get("citation")),
                abstain_enabled=abstain_enabled)
        else:
            idx, cited, failed = self._parse(content, options, valid_ids,
                                            abstain_enabled=abstain_enabled)
        if failed:
            self.stats["parse_fail"] += 1

        usage_available = (
            usage.get("prompt_tokens") is not None
            and usage.get("completion_tokens") is not None)
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        mode = f"http:{payload.get('model') or self.model_id}"
        if failed:
            mode += "+parse_fail"
        if idx < 0:
            mode += "+abstain"

        response_model = str(payload.get("model") or self.model_id)
        override = self._cost_override(prompt_tokens, completion_tokens,
                                       response_model, self.base_url)
        if not usage_available:
            override = {**override, "usd": None, "usd_nominal": None,
                        "cost_estimated": True,
                        "cost_basis": "API usage counters unavailable; cost not estimable",
                        "usage_available": False}
        else:
            override["usage_available"] = True
        if hit is None and not getattr(self, "_last_usage_complete", True):
            override = {**override, "known_success_usd": override.get("usd"),
                        "usd": None, "usd_nominal": None, "cost_estimated": True,
                        "usage_complete": False,
                        "cost_basis": "one or more dispatched attempts have unknown usage"}
        else:
            override["usage_complete"] = usage_available
        if hit is not None and usage_available:
            override = {**override, "usd": 0.0, "usd_nominal": 0.0,
                        "cost_estimated": False,
                        "cost_basis": "response cache hit; no API call charged"}
        return ModelOutput(
            answer_index=idx,
            answer_text=options[idx] if 0 <= idx < len(options) else "",
            cited_doc_ids=cited,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            mode=mode,
            raw=(content or "")[:2000],
            cost_override=override,
            provider_http_attempts=http_attempts,
            parse_failed=failed,
        )

    def _cost_override(self, p_tokens: int, c_tokens: int, model: str,
                       base_url: str | None = None) -> Dict[str, Any]:
        """Price exact API usage against a published tariff for a known endpoint.

        对齐 gepa：它把计价整个委托给 `litellm.completion_cost()`
        （`src/gepa/lm.py:133`）。本项目按供应商区域价目计价；不能把 SDK
        的通用模型 metadata 当作当前账户价格，金额没适用表就承认没有。
        """
        endpoint = base_url or self.base_url
        pricing_override = getattr(self, "pricing_override", None)
        pricing_metadata = getattr(self, "pricing_metadata", {})
        price = (dict(pricing_override)
                 if pricing_override is not None else
                 published_price_for_endpoint(model, endpoint))
        nominal = (p_tokens / 1000.0) * PRICE_PER_1K["prompt"] \
            + (c_tokens / 1000.0) * PRICE_PER_1K["completion"]
        if price is None:
            return {
                "usd": None, "cost_estimated": True,
                "cost_basis": (f"真实 token 计数（来自 API usage），没有适用于该 endpoint/model 的"
                               f"已核实价目：endpoint={endpoint}, model={model}；不输出金额"),
                "usd_nominal": round(nominal, 6),
                "prompt_tokens": p_tokens, "completion_tokens": c_tokens,
            }
        usd = (p_tokens / 1_000_000.0) * price["prompt"] \
            + (c_tokens / 1_000_000.0) * price["completion"]
        metadata = (dict(pricing_metadata) if pricing_override is not None
                    else published_tariff_metadata(model, endpoint))
        return {
            "usd": round(usd, 6), "cost_estimated": False,
            # Bind the exact four rates used by this call.  Consumers that
            # need an exact-zero proof can compare this digest with their
            # frozen plan without importing any provider/model allow-list.
            "pricing_rates_sha256": sha256_obj(price),
            "pricing_metadata": dict(metadata),
            "cost_basis": (
                str(metadata.get("pricing_formula_source") or
                    metadata.get("pricing_source") or
                    "caller-frozen tariff; not account invoice")
                if pricing_override is not None else
                (("API usage × DeepSeek official published peak upper bound tariff; "
                  f"checked={DEEPSEEK_PRICE_CHECKED_AT}; not account invoice")
                 if (urlparse(str(endpoint)).hostname or "").lower() == DEEPSEEK_API_HOST
                 else (f"API usage × SiliconFlow published list tariff; checked="
                       f"{SILICONFLOW_PRICE_CHECKED_AT}; not account invoice"))
            ),
            **metadata,
            "pricing_metadata_sha256": sha256_obj(metadata),
            "usd_nominal": round(usd, 6),
            "prompt_tokens": p_tokens, "completion_tokens": c_tokens,
        }


def build_provider(cfg: Dict[str, Any], seed: int = 0):
    kind = str(cfg.get("runtime", {}).get("provider", "fake")).lower()
    noise = float(cfg.get("runtime", {}).get("fake_noise", 0.0))
    from .channels import load_channels

    channels = load_channels(cfg)
    if kind == "fake":
        return FakeProvider(cfg.get("models", {}).get("task", {}).get("model_id", "fake-deterministic-v1"),
                            seed=seed, noise=noise, channels=channels)
    if kind == "http":
        mcfg = dict(cfg.get("models", {}).get("task", {}))
        mcfg.pop("provider", None)
        # Config-driven callers may carry an explicit tariff for a model that
        # is absent from the built-in regional table. Keep the public config
        # shape provider-neutral and translate it only at the HTTP boundary.
        if isinstance(mcfg.get("pricing"), dict):
            pricing = dict(mcfg.pop("pricing"))
            rates = {key: pricing[key] for key in
                     ("prompt", "completion", "cache_read", "cache_write")
                     if key in pricing}
            if len(rates) == 4:
                mcfg.setdefault("pricing_override", rates)
                mcfg.setdefault("pricing_metadata", pricing)
        mcfg.setdefault("model_id", "Qwen/Qwen3-8B")
        if bool((cfg.get("runtime") or {}).get("http_no_cache_required", False)):
            # Experimental A/A and intervention arms must reach the provider
            # independently. Accepting this runtime contract in the schema
            # without enforcing it here silently turned repeated arms into
            # response-cache lookups and understated model variability.
            mcfg["cache"] = False
        if (cfg.get("runtime") or {}).get("input_mode") == "task_input":
            minimum = int((cfg.get("runtime") or {}).get("task_input_max_chars", 30000))
            mcfg["max_doc_chars"] = max(int(mcfg.get("max_doc_chars", 1500)), minimum)
        return HttpProvider(channels=channels, run_seed=seed, **mcfg)
    if kind == "pi":
        from .pi_bridge import PiProvider

        return PiProvider(cfg, channels)
    raise ProviderError(f"unknown provider: {kind}")


def cost_of(output: ModelOutput) -> Dict[str, object]:
    """成本记账（P2）。

    返回三件套，语义不再含糊：
    - `usd`: API usage × 适用的已发布供应商价目；不是账户账单。
      **离线确定性模型下恒为 None**（没有真实计价可言）。
    - `cost_estimated`: 是否为估算。
    - `cost_basis`: 估算方法说明，写进报告。

    真实 provider 走 `ModelOutput.cost_override`（真 token + endpoint/model适用的公开单价）；
    离线 fake 走名义估算。两者都不会把"估的"冒充成"账单"。

    对齐 gepa：它对 "callable LM" 明确声明 `total_cost = 0.0`、token 按
    `len(text)//4` 估算（`src/gepa/lm.py:216-218` 与
    `docs/docs/guides/cost-tracking.md`）。本项目在它基础上更保守一步 ——
    **不填 0.0 而填 None**，避免读者把 0 误读成"没花钱"。
    """
    if output.cost_override:
        return dict(output.cost_override)
    nominal = (output.prompt_tokens / 1000.0) * PRICE_PER_1K["prompt"] \
        + (output.completion_tokens / 1000.0) * PRICE_PER_1K["completion"]
    return {
        "usd": None,
        "cost_estimated": True,
        "cost_basis": COST_BASIS,
        # 名义估算值单独存放，仅供量级参考，**不要当作成本结论引用**
        "usd_nominal": round(nominal, 6),
        "prompt_tokens": output.prompt_tokens,
        "completion_tokens": output.completion_tokens,
    }
