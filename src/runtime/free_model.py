"""Verify a current official zero-price entry before engineering API tests."""
from __future__ import annotations
import json
import re
from decimal import Decimal
import requests
from ..util import now_iso, sha256_file, write_json

PRICE_URL = "https://www.siliconflow.cn/pricing"


def parse_zero_price(html, model_id):
    frames = [json.loads(value) for value in re.findall(
        r'self\.__next_f\.push\(\[1,("(?:\\.|[^"\\])*")\]\)', html)]
    objects = {}
    for line in "".join(frames).splitlines():
        key, separator, value = line.partition(":")
        if separator:
            try:
                objects[key] = json.loads(value)
            except ValueError:
                pass
    models = [value for value in objects.values() if isinstance(value, dict) and value.get("modelName") == model_id]
    if len(models) != 1 or models[0].get("status") != "normal":
        raise ValueError("official pricing page does not identify one active requested model")
    entries = objects.get(str(models[0].get("pricing", "")).removeprefix("$"))
    if not isinstance(entries, list):
        raise ValueError("official pricing schema changed; zero-price test is blocked")
    prices = {}
    for reference in entries:
        item = objects.get(str(reference).removeprefix("$"))
        if isinstance(item, dict):
            prices[item.get("specification")] = item.get("price")
    if not {"prompt", "completion"} <= set(prices) or any(Decimal(str(prices[k])) != 0 for k in ("prompt", "completion")):
        raise ValueError("requested model is not currently zero-priced for both input and output")
    return {"model_id": model_id, "prompt_price": prices["prompt"], "completion_price": prices["completion"]}


def verify_current(model_id, directory):
    directory.mkdir(parents=True, exist_ok=False)
    response = requests.get(PRICE_URL, timeout=20, stream=True, allow_redirects=False)
    try:
        if response.status_code != 200:
            raise ValueError("official pricing page unavailable; no model request authorized")
        payload = bytearray()
        for block in response.iter_content(65536):
            if len(payload) + len(block) > 2_000_000:
                raise ValueError("pricing page exceeded verification byte ceiling")
            payload.extend(block)
        result = parse_zero_price(payload.decode("utf-8"), model_id)
        page = directory / "official-pricing.html"
        page.write_bytes(payload)
        receipt = {"schema": "official-free-model-check-v1", "source": PRICE_URL,
                   "retrieved_at": now_iso(), "source_sha256": sha256_file(page),
                   "max_model_cost_usd": 0, **result}
        write_json(directory / "receipt.json", receipt)
        return receipt
    finally:
        response.close()
