#!/usr/bin/env python
"""为 SuppQA 抓取论文全文（R0 的一环）。

背景
    SuppQA 的 parquet 里**只有 paper-title + DOI，没有任何正文**，
    而它是唯一允许做最终结论的封存集。此前它能答题全靠 gateway 注入的
    `Reported finding: <答案>` —— 那是"答案直给"，不是检索。
    本脚本用 DOI 走 OpenAlex 找开放获取全文，下载 PDF 抽取文本，
    落盘到 `data/cache/suppqa_fulltext/`（**不入库，版权原因**）。

用法
    python scripts/fetch_suppqa_fulltext.py            # 全量
    python scripts/fetch_suppqa_fulltext.py --limit 5  # 试跑

产物
    data/cache/suppqa_fulltext/<sha1(doi)>.json
      {"doi","title","pdf_url","is_oa","ok","text_len","text","reason"}
    索引：data/cache/suppqa_fulltext/index.json
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import time
from pathlib import Path
from typing import Any, Dict

import requests

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data" / "cache" / "suppqa_fulltext"
PARQUET = REPO / "vendor" / "SuppQA" / "train-00000-of-00001.parquet"
UA = "rsi-selfevolve/1.0 (research benchmark corpus build; mailto:local@example.com)"
MIN_TEXT = 400          # 少于 400 字符视为抓取失败（多半是扫描件/仅封面）
MIN_ABSTRACT = 300      # 摘要回退的门槛（OpenAlex 摘要通常 800+ 字符）
MAX_TEXT = 60000        # 截断，避免语料过大


def _inverted_to_text(inv: Any) -> str:
    """OpenAlex 的 abstract_inverted_index（{词: [位置...]}）还原成原文。"""
    if not isinstance(inv, dict) or not inv:
        return ""
    slots: list[str] = []
    for word, positions in inv.items():
        for p in positions or []:
            while len(slots) <= p:
                slots.append("")
            slots[p] = word
    return " ".join(w for w in slots if w)


def _epmc_lookup(doi: str, sess: requests.Session) -> Dict[str, Any]:
    """Europe PMC：生物医学论文的摘要覆盖率约 93%（实测 15 篇里 14 篇）。

    OpenAlex 对本批文献几乎不给 abstract_inverted_index（82 篇只多救回 1 篇），
    所以必须走 EPMC。优先拿 PMC 开放全文，其次摘要。
    """
    out: Dict[str, Any] = {}
    try:
        r = sess.get("https://www.ebi.ac.uk/europepmc/webservices/rest/search",
                     params={"query": f'DOI:"{doi}"', "resultType": "core",
                             "format": "json"}, timeout=25)
        if r.status_code != 200:
            return out
        res = (r.json().get("resultList") or {}).get("result") or []
        if not res:
            return out
        d = res[0]
        if len(d.get("abstractText") or "") >= MIN_ABSTRACT:
            out["abstract"] = d["abstractText"]
        if d.get("pmcid") and d.get("isOpenAccess") == "Y":
            try:
                fx = sess.get(
                    f"https://www.ebi.ac.uk/europepmc/webservices/rest/"
                    f"{d['pmcid']}/fullTextXML", timeout=40)
                if fx.status_code == 200 and len(fx.text) > 2000:
                    txt = re.sub(r"<[^>]+>", " ", fx.text)
                    txt = _clean(re.sub(r"\s+", " ", txt))
                    if len(txt) >= MIN_TEXT:
                        out["fulltext"] = txt
                        out["fulltext_url"] = f"europepmc:{d['pmcid']}"
            except Exception:
                pass
    except Exception:
        return out
    return out


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


def _pdf_to_text(data: bytes) -> str:
    """依次尝试 pypdf → pdfminer → PyMuPDF，谁先成功用谁。"""
    # 1) pypdf
    try:
        from pypdf import PdfReader
        r = PdfReader(io.BytesIO(data))
        t = "\n".join((p.extract_text() or "") for p in r.pages[:40])
        if len(t.strip()) >= MIN_TEXT:
            return t.strip()[:MAX_TEXT]
    except Exception:
        pass
    # 2) pdfminer
    try:
        from pdfminer.high_level import extract_text
        t = extract_text(io.BytesIO(data), maxpages=40)
        if len(t.strip()) >= MIN_TEXT:
            return t.strip()[:MAX_TEXT]
    except Exception:
        pass
    # 3) PyMuPDF
    try:
        import fitz
        doc = fitz.open(stream=data, filetype="pdf")
        t = "\n".join(doc[i].get_text() for i in range(min(40, doc.page_count)))
        if len(t.strip()) >= MIN_TEXT:
            return t.strip()[:MAX_TEXT]
    except Exception:
        pass
    return ""


def _clean(t: str) -> str:
    t = t.replace("\x00", " ")
    t = "\n".join(ln.strip() for ln in t.splitlines() if ln.strip())
    return t[:MAX_TEXT]


def main() -> int:
    raise SystemExit("retired_unplanned_entrypoint: the historical SuppQA fulltext fetcher has no frozen development cohort or request/response budget. Use scripts/research_assistant.py lookup/discover for bounded DOI metadata and abstract retrieval.")
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=0.3)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(PARQUET)
    rows = df.to_dict("records")
    if args.limit:
        rows = rows[: args.limit]

    sess = requests.Session()
    sess.headers.update({"User-Agent": UA})
    index: Dict[str, Any] = {}
    n_ok, n_oa, n_pdf, n_abs = 0, 0, 0, 0

    for i, r in enumerate(rows, 1):
        doi = str(r.get("source", "")).replace("https://doi.org/", "").strip()
        title = str(r.get("paper-title", "")).strip()
        key = _sha1(doi or title)
        fp = OUT / f"{key}.json"
        rec: Dict[str, Any] = {"doi": doi, "title": title, "ok": False}

        if fp.exists():
            try:
                rec = json.loads(fp.read_text(encoding="utf-8"))
            except Exception:
                rec = {"doi": doi, "title": title, "ok": False}
        # 只有"已拿到全文"才跳过网络请求；失败的和只拿到摘要的重跑，尝试升级
        already_full = bool(rec.get("ok")) and rec.get("kind") == "fulltext"

        if not already_full:
            pdf_url = None
            try:
                resp = sess.get(f"https://api.openalex.org/works/doi:{doi}", timeout=30)
                if resp.status_code == 200:
                    d = resp.json()
                    rec["is_oa"] = bool((d.get("open_access") or {}).get("is_oa"))
                    n_oa += bool(rec["is_oa"])
                    # 摘要回退：全文抓不到时用它（OpenAlex 覆盖率远高于可解析 PDF）
                    abst = _inverted_to_text(d.get("abstract_inverted_index"))
                    if len(abst) >= MIN_ABSTRACT:
                        rec["abstract"] = abst
                    locs = d.get("locations") or []
                    best = d.get("best_oa_location") or {}
                    pdf_url = best.get("pdf_url")
                    if not pdf_url:
                        for loc in locs:
                            if loc.get("pdf_url"):
                                pdf_url = loc["pdf_url"]
                                break
                else:
                    rec["reason"] = f"openalex_http_{resp.status_code}"
            except Exception as exc:
                rec["reason"] = f"openalex_{type(exc).__name__}"

            if pdf_url:
                n_pdf += 1
                try:
                    pr = sess.get(pdf_url, timeout=60,
                                  headers={"User-Agent": UA, "Accept": "application/pdf,*/*"})
                    ct = (pr.headers.get("Content-Type") or "").lower()
                    if pr.status_code == 200 and len(pr.content) > 3000:
                        if "pdf" in ct or pr.content[:4] == b"%PDF":
                            txt = _clean(_pdf_to_text(pr.content))
                        else:
                            txt = _clean(pr.text)
                        if len(txt) >= MIN_TEXT:
                            rec.update(ok=True, pdf_url=pdf_url,
                                       text_len=len(txt), text=txt)
                            n_ok += 1
                        else:
                            rec["reason"] = f"too_short_{len(txt)}"
                    else:
                        rec["reason"] = f"pdf_http_{pr.status_code}_len{len(pr.content)}"
                except Exception as exc:
                    rec["reason"] = f"download_{type(exc).__name__}"
            elif "reason" not in rec:
                rec["reason"] = "no_pdf_url"

            # 回退链：EPMC 开放全文 → EPMC 摘要 → OpenAlex 摘要。
            # 三者都是真实论文文本（不是我们编的），但信息量递减，
            # 用 kind 标出来，评测报告里要如实写。
            if not rec.get("ok"):
                e = _epmc_lookup(doi, sess)
                if e.get("fulltext"):
                    rec.update(ok=True, kind="fulltext", text=e["fulltext"],
                               text_len=len(e["fulltext"]),
                               pdf_url=e.get("fulltext_url", ""), reason="")
                    n_ok += 1
                elif len(e.get("abstract", "")) >= MIN_ABSTRACT:
                    rec.update(ok=True, kind="abstract-epmc", text=e["abstract"],
                               text_len=len(e["abstract"]), reason="")
                    n_abs += 1
                    n_ok += 1
                elif len(rec.get("abstract", "")) >= MIN_ABSTRACT:
                    rec.update(ok=True, kind="abstract", text=rec["abstract"],
                               text_len=len(rec["abstract"]), reason="")
                    n_abs += 1
                    n_ok += 1
            if rec.get("ok") and not rec.get("kind"):
                rec["kind"] = "fulltext"

        fp.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
        index[key] = {"doi": doi, "ok": bool(rec.get("ok")),
                      "text_len": int(rec.get("text_len", 0)),
                      "reason": rec.get("reason", "")}
        if i % 10 == 0 or i == len(rows):
            print(f"[{i}/{len(rows)}] ok={n_ok} (全文+摘要) pdf={n_pdf} 摘要回退={n_abs}", flush=True)
        time.sleep(args.sleep)

    (OUT / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
    # 最终计数以**落盘文件**为准：重跑时内存计数会漏掉"上一轮已成功"的条目，
    # 之前因此把覆盖率误报成 63%（实际 81/82）。
    n_ok = n_ft = n_ab = 0
    for fp in OUT.glob("*.json"):
        if fp.name == "index.json":
            continue
        try:
            d = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        if d.get("ok"):
            n_ok += 1
            if d.get("kind") == "fulltext":
                n_ft += 1
            else:
                n_ab += 1
    tot = len(rows)
    print(f"=== 完成 {tot} 篇：可用文本 {n_ok} ({n_ok/max(1,tot):.0%})，"
          f"其中全文 {n_ft}、摘要 {n_ab}；OA {n_oa}，有 PDF 直链 {n_pdf} ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
