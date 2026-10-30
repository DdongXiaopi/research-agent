"""Train-only human evidence-document retrieval benchmark for BM25 variants.

Uses SciFact train claims with annotated evidence, never SciFact dev or BioASQ
confirmation. The hash-held gate portion is reported once for fixed variants;
it is not a candidate-generation set.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.retrieval.pubmed_snapshot import _STOP as PRODUCTION_STOP
from src.retrieval.pubmed_snapshot import _WORD_V7 as WORD


MINIMAL_STOP = {"are", "can", "could", "does", "do", "did", "has", "have", "had",
        "how", "is", "it", "may", "might", "of", "the", "this", "that",
        "there", "these", "those", "what", "when", "where", "which",
        "who", "why", "with", "without", "for", "from", "into", "and",
        "or", "not", "only", "been", "being", "was", "were", "than", "more"}


def words(text: str, stop: set[str] | None = None) -> list[str]:
    return [w.lower() for w in WORD.findall(text)
            if stop is None or w.lower() not in stop]


class Index:
    def __init__(self, corpus: list[dict]):
        self.ids = [str(row["doc_id"]) for row in corpus]
        self.contents = []
        self.postings = [defaultdict(list), defaultdict(list)]
        self.lengths = [[], []]
        for i, row in enumerate(corpus):
            abstract = row["abstract"]
            if isinstance(abstract, str) and abstract.startswith("["):
                abstract = ast.literal_eval(abstract)
            abstract_text = " ".join(abstract) if isinstance(abstract, list) else str(abstract)
            title = str(row["title"])
            self.contents.append(title + " " + abstract_text)
            for field, content in enumerate((title + " " + abstract_text, title)):
                counts = Counter(words(content))
                self.lengths[field].append(sum(counts.values()))
                for term, tf in counts.items():
                    self.postings[field][term].append((i, tf))
        self.avg = [sum(lengths) / len(lengths) for lengths in self.lengths]

    def rank(self, query: str, title_weight: float, stop: set[str],
             k: int = 10) -> list[str]:
        terms = list(dict.fromkeys(words(query, stop)))
        scores = defaultdict(float)
        n_docs = len(self.ids)
        for field, weight in ((0, 1.0), (1, title_weight)):
            if weight == 0:
                continue
            for term in terms:
                postings = self.postings[field].get(term, [])
                df = len(postings)
                if not df:
                    continue
                idf = math.log1p((n_docs - df + 0.5) / (df + 0.5))
                for i, tf in postings:
                    size = self.lengths[field][i]
                    scores[i] += (weight * idf * tf * 2.2 /
                                  (tf + 1.2 * (0.25 + 0.75 * size / self.avg[field])))
        ranked = sorted(scores, key=lambda i: (-scores[i], self.ids[i]))[:k]
        return [self.ids[i] for i in ranked]


def summarize(rows: list[dict], ranks: dict[str, list[str]]) -> dict:
    out = {}
    for k in (1, 3, 5, 10):
        out[f"recall_at_{k}"] = round(sum(bool(set(r["evidence"]) &
            set(ranks[str(r["id"])][:k])) for r in rows) / len(rows), 6)
    out["mrr_at_10"] = round(sum(next((1.0 / (i + 1) for i, doc in
        enumerate(ranks[str(r["id"])]) if doc in r["evidence"]), 0.0)
        for r in rows) / len(rows), 6)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path,
                        default=Path("artifacts/probe/scifact/data"))
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/probe/scifact/retrieval-train-bm25.json"))
    parser.add_argument("--char-probe", action="store_true",
                        help="Compare character TF-IDF and fixed RRF with BM25")
    args = parser.parse_args()
    corpus = [json.loads(line) for line in (args.data / "corpus.jsonl").open(
        encoding="utf-8")]
    claims = [json.loads(line) for line in (args.data / "claims_train.jsonl").open(
        encoding="utf-8")]
    claims = [row for row in claims if row["evidence"]]
    by_id = {str(row["doc_id"]) for row in corpus}
    if not claims or any(not set(row["evidence"]) <= by_id for row in claims):
        raise ValueError("SciFact train evidence IDs missing from corpus")
    optimize = [row for row in claims if int(hashlib.sha256(
        str(row["id"]).encode()).hexdigest(), 16) % 2 == 0]
    gate = [row for row in claims if row not in optimize]
    index = Index(corpus)
    variants = {}
    for name, weight, stop in (
            ("production_bm25", 0.0, PRODUCTION_STOP),
            ("minimal_stop_bm25", 0.0, MINIMAL_STOP),
            ("production_title_0.5", 0.5, PRODUCTION_STOP),
            ("production_title_1.0", 1.0, PRODUCTION_STOP)):
        ranks = {str(row["id"]): index.rank(row["claim"], weight, stop)
                 for row in claims}
        variants[name] = {
            "optimize": summarize(optimize, ranks),
            "gate": summarize(gate, ranks),
            "gate_failures": [row["id"] for row in gate
                              if not set(row["evidence"]) &
                              set(ranks[str(row["id"])][:3])],
        }
    if args.char_probe:
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                     min_df=2, sublinear_tf=True)
        doc_matrix = vectorizer.fit_transform(index.contents)
        claim_matrix = vectorizer.transform([row["claim"] for row in claims])
        char_ranks = {}
        fused_ranks = {}
        for pos, row in enumerate(claims):
            claim_id = str(row["id"])
            scores = (claim_matrix[pos] @ doc_matrix.T).toarray().ravel()
            order = np.argsort(-scores, kind="mergesort")[:100]
            char_top100 = [index.ids[int(i)] for i in order]
            bm25_top100 = index.rank(row["claim"], 0.0,
                                    PRODUCTION_STOP, k=100)
            char_ranks[claim_id] = char_top100[:10]
            fused = defaultdict(float)
            for ranking in (char_top100, bm25_top100):
                for rank, doc_id in enumerate(ranking):
                    fused[doc_id] += 1.0 / (60 + rank + 1)
            fused_ranks[claim_id] = sorted(
                fused, key=lambda doc_id: (-fused[doc_id], doc_id))[:10]
        for name, ranks in (("char_tfidf", char_ranks),
                            ("bm25_char_rrf", fused_ranks)):
            variants[name] = {
                "optimize": summarize(optimize, ranks),
                "gate": summarize(gate, ranks),
                "gate_failures": [row["id"] for row in gate
                                  if not set(row["evidence"]) &
                                  set(ranks[str(row["id"])][:3])],
            }
    report = {"status": "scifact_train_only_retrieval_benchmark",
              "source": "official SciFact corpus and claims_train.jsonl",
              "n_corpus": len(corpus), "n_labeled_claims": len(claims),
              "n_optimize": len(optimize), "n_gate": len(gate),
              "split": "SHA256(claim ID) parity; dev not read",
              "variants": variants,
              "limitation": "SciFact evidence-document recall does not prove BioASQ "
                            "answer accuracy or citation grounding."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(json.dumps({"n_labeled_claims": len(claims),
                      "n_optimize": len(optimize), "n_gate": len(gate),
                      "variants": {name: {split: {k: v for k, v in val[split].items()
                                                  if k != "gate_failures"}
                                          for split in ("optimize", "gate")}
                                   for name, val in variants.items()}}, indent=2))


if __name__ == "__main__":
    main()
