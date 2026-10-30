"""Shared bounded data contract; copied into the immutable rendering image."""
import math

SCHEMA = "rsi-render-worker-v1"
MAX_REQUEST_BYTES = 300000
MAX_OUTPUT_BYTES = 16 * 1024 * 1024
MAX_ARTIFACT_BYTES = 8 * 1024 * 1024
MAX_TOTAL_BYTES = 10 * 1024 * 1024


def validate_request(kind, payload):
    if not isinstance(payload, dict):
        raise ValueError("worker payload must be an object")
    if kind == "latex_compile":
        if set(payload) != {"source"} or not isinstance(payload["source"], str) or not 1 <= len(payload["source"]) <= 40000:
            raise ValueError("LaTeX worker requires source text of 1..40000 characters")
        return dict(payload)
    if kind == "selftest" and not payload:
        return {}
    if kind != "plot" or set(payload) - {"series", "title", "xlabel", "ylabel", "style", "formats", "dpi"}:
        raise ValueError("unsupported rendering request; Python and shell are not accepted")
    series = payload.get("series")
    if not isinstance(series, list) or not 1 <= len(series) <= 8:
        raise ValueError("plot requires 1..8 numeric series")
    for item in series:
        if not isinstance(item, dict) or set(item) != {"label", "x", "y"} or not isinstance(item["label"], str) or len(item["label"]) > 120:
            raise ValueError("invalid plot series fields")
        if (not isinstance(item["x"], list) or not isinstance(item["y"], list)
                or not 1 <= len(item["x"]) <= 2000 or len(item["x"]) != len(item["y"])):
            raise ValueError("plot series must have 1..2000 paired numeric values")
        if any(type(v) not in (int, float) or not math.isfinite(v) or abs(v) > 1e100 for v in item["x"] + item["y"]):
            raise ValueError("plot values must be finite bounded numbers")
    for key in ("title", "xlabel", "ylabel"):
        if not isinstance(payload.get(key, ""), str) or len(payload.get(key, "")) > 240:
            raise ValueError("plot label exceeds text budget")
    formats = payload.get("formats", ["png", "pdf", "svg"])
    if (not isinstance(formats, list) or not formats or len(formats) != len(set(formats))
            or any(v not in {"png", "pdf", "svg"} for v in formats)):
        raise ValueError("plot formats must be unique png/pdf/svg values")
    if payload.get("style", "line") not in {"line", "scatter", "bar"}:
        raise ValueError("unsupported plot style")
    if type(payload.get("dpi", 160)) is not int or not 72 <= payload.get("dpi", 160) <= 300:
        raise ValueError("plot DPI must be an integer from 72 to 300")
    return {**payload, "formats": formats, "dpi": payload.get("dpi", 160), "style": payload.get("style", "line")}
