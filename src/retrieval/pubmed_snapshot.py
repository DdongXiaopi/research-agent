"""Frozen PubMed article candidates for a new, train-first retrieval protocol.

The builder uses NCBI ESearch/EFetch only while creating a snapshot. A runner
never calls NCBI: both experimental arms rank the same saved articles. No gold
answer or BioASQ supplied passage is stored in the snapshot.
"""

from __future__ import annotations

import math
import re
import time
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests

from ..util import sha256_obj, sha256_text
from .gateway import RetrievalError, RetrievalGateway

BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PROTOCOL = "pubmed-esearch-efetch-snapshot-v7"
DOTTED_PROTOCOLS = {"pubmed-esearch-efetch-snapshot-v5",
                    "pubmed-esearch-efetch-snapshot-v6", PROTOCOL}
SUPPORTED_PROTOCOLS = {"pubmed-esearch-efetch-snapshot-v1",
                       "pubmed-esearch-efetch-snapshot-v2",
                       "pubmed-esearch-efetch-snapshot-v3",
                       "pubmed-esearch-efetch-snapshot-v4",
                       *DOTTED_PROTOCOLS}
# Keep dotted biomedical names such as H2A.Z intact for PubMed ESearch.
# The v4 tokenizer silently reduced this entity to H2A and pulled unrelated
# Ino80 abstracts into an RSC question's candidate pool.
_WORD_V4 = re.compile(r"[a-z0-9][a-z0-9-]{2,}", re.I)
_WORD = re.compile(r"[a-z0-9][a-z0-9-]{2,}(?:\.[a-z0-9]+)*", re.I)
_WORD_V7 = re.compile(
    r"[a-z0-9\u0370-\u03ff][a-z0-9\u0370-\u03ff-]{2,}"
    r"(?:\.[a-z0-9\u0370-\u03ff]+)*", re.I)
_STOP = {"are", "can", "could", "does", "do", "did", "has", "have", "had",
         "how", "is", "it", "may", "might", "of", "the", "this", "that",
         "there", "these", "those", "what", "when", "where", "which",
         "who", "why", "with", "without", "for", "from", "into", "and",
         "or", "not", "only", "been", "being", "was", "were", "than", "more",
         # Generic question framing narrows PubMed's implicit AND search but
         # rarely names the biomedical entity. Keep specific entity terms.
         "affect", "associated", "association", "cause", "caused", "clinical",
         "data", "effect", "effective", "evidence", "possible", "result",
         "results", "risk", "role", "study", "studies", "support", "trial",
         "trials", "treatment", "use", "used", "between", "link", "levels",
         "decrease", "upon", "phenotype", "present", "following",
         "play", "plays"}


def keyword_query(question: str, max_terms: int = 8,
                  word_pattern: re.Pattern = _WORD) -> str:
    """Label-free, deterministic broad query to widen the snapshot pool."""
    terms = [w.lower() for w in word_pattern.findall(question)
             if w.lower() not in _STOP]
    return " ".join(dict.fromkeys(terms[:max_terms])) or question


def entity_pair_query(question: str, word_pattern: re.Pattern = _WORD_V7) -> str:
    """Query two distinctive biomedical names without generic AND terms."""
    words = [(i, w) for i, w in enumerate(word_pattern.findall(question))
             if w.lower() not in _STOP]
    distinctive = [(i, w) for i, w in words
                   if not re.fullmatch(r"\d+-item", w, re.I)
                   and (("." in w or any(c.isdigit() for c in w))
                        or (w.isupper() and len(w) >= 2
                            and w not in {"DNA", "RNA", "HRQOL", "CRS"}))]
    if len(distinctive) >= 2:
        picked = distinctive[:2]
    elif len(distinctive) == 1:
        # Only a nearby *preceding* proper name can complete a coded target.
        # A later title-cased generic phrase (for example Health-Related)
        # would turn a measurement synonym into a false entity pair.
        target_index = distinctive[0][0]
        candidates = [(i, w) for i, w in words
                      if 0 < target_index - i <= 3 and w[:1].isupper()
                      and not w.isupper() and len(w) >= 4
                      and "-" not in w]
        if not candidates:
            return ""
        picked = [distinctive[0], candidates[0]]
    else:
        return ""
    return " ".join(w for _, w in sorted(picked))


def fallback_query(question: str,
                   word_pattern: re.Pattern = _WORD_V7) -> str:
    """Broad two-entity query when the first two searches yield no abstracts."""
    words = list(dict.fromkeys(w.lower() for w in word_pattern.findall(question)
                               if w.lower() not in _STOP))
    if not words:
        return question
    coded = [i for i, word in enumerate(words) if any(ch.isdigit() for ch in word)]
    strongest = [max(coded, key=lambda i: (len(words[i]), -i))] if coded else []
    strongest.extend(i for i in sorted(range(len(words)),
                                         key=lambda i: (-len(words[i]), i))
                     if i not in strongest)
    strongest = strongest[:2]
    return " ".join(words[i] for i in sorted(strongest)) or question


def broad_query(question: str,
                word_pattern: re.Pattern = _WORD_V7) -> str:
    """Relax one restrictive term to expand a sparse, label-free pool."""
    words = list(dict.fromkeys(w.lower() for w in word_pattern.findall(question)
                               if w.lower() not in _STOP))
    if len(words) < 3:
        return fallback_query(question, word_pattern=word_pattern)
    longest = max(range(len(words)), key=lambda i: (len(words[i]), -i))
    return " ".join(word for i, word in enumerate(words) if i != longest)


def _text(element: Optional[ET.Element]) -> str:
    return " ".join(" ".join(element.itertext()).split()) if element is not None else ""


def parse_pubmed_xml(xml: bytes) -> Dict[str, Dict[str, str]]:
    root = ET.fromstring(xml)
    out: Dict[str, Dict[str, str]] = {}
    for article in root.findall(".//PubmedArticle"):
        pmid = _text(article.find("./MedlineCitation/PMID"))
        title = _text(article.find("./MedlineCitation/Article/ArticleTitle"))
        abstracts = [_text(a) for a in article.findall(
            "./MedlineCitation/Article/Abstract/AbstractText")]
        abstract = " ".join(x for x in abstracts if x)
        if not pmid or not title or len(abstract) < 80:
            continue
        doi = ""
        for aid in article.findall("./PubmedData/ArticleIdList/ArticleId"):
            if aid.get("IdType") == "doi":
                doi = _text(aid)
                break
        out[pmid] = {"doc_id": "pmid:" + pmid, "pmid": pmid,
                     "title": title, "text": title + "\n" + abstract,
                     "doi": doi, "url": "https://pubmed.ncbi.nlm.nih.gov/" + pmid + "/"}
    return out


class PubMedClient:
    def __init__(self, session: Optional[requests.Session] = None,
                 min_interval: float = 0.4):
        self.session = session or requests.Session()
        self.min_interval = min_interval
        self._last_request = 0.0

    def _get(self, endpoint: str, params: Dict[str, Any]) -> requests.Response:
        for attempt in range(3):
            delay = self.min_interval - (time.monotonic() - self._last_request)
            if delay > 0:
                time.sleep(delay)
            self._last_request = time.monotonic()
            try:
                response = self.session.get(
                    BASE_URL + "/" + endpoint,
                    params={**params, "tool": "rsi-research-prototype"}, timeout=25)
                if response.status_code in {429, 500, 502, 503, 504}:
                    if attempt < 2:
                        time.sleep(1 + attempt * 2)
                        continue
                response.raise_for_status()
                return response
            except requests.RequestException:
                if attempt == 2:
                    raise
                time.sleep(1 + attempt * 2)
        raise RuntimeError("unreachable PubMed retry state")

    def search(self, query: str, limit: int = 12) -> List[str]:
        response = self._get("esearch.fcgi", {
            "db": "pubmed", "term": query, "retmax": limit,
            "retmode": "json", "sort": "relevance"})
        return [str(x) for x in response.json()["esearchresult"]["idlist"]]

    def fetch(self, pmids: List[str]) -> Dict[str, Dict[str, str]]:
        if not pmids:
            return {}
        response = self._get("efetch.fcgi", {
            "db": "pubmed", "id": ",".join(pmids), "retmode": "xml"})
        return parse_pubmed_xml(response.content)


def build_train_snapshot(tasks: List[Dict[str, Any]], client: PubMedClient,
                         per_query: int = 12) -> Dict[str, Any]:
    """Build from public train only; never read labels or supplied evidence."""
    if not tasks or any(t.get("dataset") != "bioasq_unitor" or
                        t.get("split") != "dev_optimize" for t in tasks):
        raise ValueError("PubMed snapshot builder accepts BioASQ train tasks only")
    allowed = {"task_id", "dataset", "split", "question"}
    if any(set(t) != allowed for t in tasks):
        raise ValueError("PubMed snapshot builder accepts question-only rows")
    rows = []
    for task in tasks:
        question = str(task["question"])
        queries = {"raw": question,
                   "keywords": keyword_query(question,
                                             word_pattern=_WORD_V7)}
        entity_pair = entity_pair_query(question, word_pattern=_WORD_V7)
        if entity_pair:
            queries["entity_pair"] = entity_pair
        hits = {name: client.search(query, per_query)
                for name, query in queries.items()}
        pmids = list(dict.fromkeys(pmid for name in queries for pmid in hits[name]))
        fetched = client.fetch(pmids)
        # PubMed implicitly ANDs many terms. Sparse pools can exclude later
        # trials even when they are relevant; broaden without using labels.
        if len(fetched) < 10:
            queries["broad"] = broad_query(question, word_pattern=_WORD_V7)
            hits["broad"] = client.search(queries["broad"], per_query)
            more = [p for p in hits["broad"] if p not in pmids]
            pmids.extend(more)
            fetched.update(client.fetch(more))
        if len(fetched) < 10:
            queries["fallback"] = fallback_query(question,
                                                 word_pattern=_WORD_V7)
            hits["fallback"] = client.search(queries["fallback"], per_query)
            more = [p for p in hits["fallback"] if p not in pmids]
            pmids.extend(more)
            fetched.update(client.fetch(more))
        docs = [fetched[pmid] for pmid in pmids if pmid in fetched]
        rows.append({"task_id": task["task_id"],
                     "question_sha256": sha256_text(question),
                     "queries": queries, "pmids_by_query": hits,
                     "docs": docs})
    obj = {"protocol": PROTOCOL,
           "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
           "source": "NCBI PubMed ESearch relevance + EFetch XML",
           "per_query": per_query, "tasks": rows}
    obj["content_sha256"] = sha256_obj(obj)
    return obj


class PubMedSnapshotGateway(RetrievalGateway):
    """BM25 ranker over the identical saved PubMed articles for both arms."""

    def __init__(self, snapshot: Dict[str, Any], top_k: int = 3,
                 passage_mode: str = "full"):
        super().__init__(backend="offline-fixture", top_k=top_k)
        if passage_mode not in {"full", "query_window"}:
            raise RetrievalError("unsupported PubMed passage mode")
        self.passage_mode = passage_mode
        if snapshot.get("protocol") not in SUPPORTED_PROTOCOLS:
            raise RetrievalError("unsupported PubMed snapshot protocol")
        self.snapshot_protocol = str(snapshot["protocol"])
        self.word_pattern = (_WORD_V7 if self.snapshot_protocol == PROTOCOL else
                             _WORD if self.snapshot_protocol in DOTTED_PROTOCOLS
                             else _WORD_V4)
        checksum = snapshot.get("content_sha256")
        body = {k: v for k, v in snapshot.items() if k != "content_sha256"}
        if checksum != sha256_obj(body):
            raise RetrievalError("PubMed snapshot checksum mismatch")
        self.content_sha256 = checksum
        self.rows = {r["task_id"]: r for r in snapshot["tasks"]}
        if len(self.rows) != len(snapshot["tasks"]):
            raise RetrievalError("duplicate task IDs in PubMed snapshot")

    def corpus(self, task: Dict[str, Any]) -> List[Dict[str, Any]]:
        row = self.rows.get(task["task_id"])
        if row is None or row["question_sha256"] != sha256_text(task["question"]):
            raise RetrievalError("task absent from PubMed snapshot or question changed")
        return [dict(x) for x in row["docs"]]

    def corpus_for_query_names(self, task: Dict[str, Any],
                               names: List[str]) -> List[Dict[str, Any]]:
        """Replay only the papers captured by named ESearch calls."""
        docs = self.corpus(task)
        if (not isinstance(names, list) or not names
                or any(not isinstance(name, str) or not name for name in names)
                or len(names) != len(set(names))):
            raise RetrievalError("PubMed source query names must be unique nonempty strings")
        row = self.rows[task["task_id"]]
        hits = row.get("pmids_by_query") or {}
        if any(name not in hits for name in names):
            raise RetrievalError("requested PubMed source query is absent from snapshot")
        pmids = list(dict.fromkeys(pmid for name in names for pmid in hits[name]))
        allowed = {"pmid:" + pmid for pmid in pmids}
        return [doc for doc in docs if doc["doc_id"] in allowed]

    def available_query_names(self, task: Dict[str, Any]) -> List[str]:
        self.corpus(task)  # Validate task identity and question hash.
        return list((self.rows[task["task_id"]].get("pmids_by_query") or {}).keys())

    def available_source_queries(self, task: Dict[str, Any]) -> Dict[str, str]:
        """Expose only saved question-derived ESearch strings, never hits or labels."""
        names = self.available_query_names(task)
        queries = self.rows[task["task_id"]].get("queries") or {}
        if any(name not in queries for name in names):
            raise RetrievalError("captured source query text is missing")
        return {name: str(queries[name]) for name in names}

    def anchor_query(self, question: str, max_terms: int = 20) -> str:
        """Keep a frozen snapshot's original tokenization on replay."""
        return keyword_query(question, max_terms=max_terms,
                             word_pattern=self.word_pattern)

    def evidence_signal(self, task: Dict[str, Any], text: str) -> bool:
        """Topical relevance proxy for yes/no abstracts, never an answer oracle.

        The fixture's option-overlap heuristic treats the word 'yes' as
        evidence and cannot handle an absent 'no' token. A PubMed passage must
        instead overlap at least two non-stopword question terms. This does
        not assert that the passage supports the correct answer.
        """
        terms = {w.lower() for w in self.word_pattern.findall(task["question"])
                 if w.lower() not in _STOP}
        passage = {w.lower() for w in self.word_pattern.findall(text)}
        return len(terms & passage) >= 2

    def search(self, task: Dict[str, Any], skill_recalled: bool = False,
               rerank: bool = False, query: Optional[str] = None,
               corpus: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        docs = [dict(x) for x in (corpus if corpus is not None else self.corpus(task))]
        if not docs:
            return []
        actual_query = query if query is not None else task["question"]
        if rerank:
            actual_query += " " + " ".join(task.get("options", []))
        qterms = list(dict.fromkeys(w.lower() for w in self.word_pattern.findall(actual_query)
                                    if w.lower() not in _STOP))
        doc_terms = [Counter(w.lower() for w in self.word_pattern.findall(d["text"]))
                     for d in docs]
        avg_len = sum(sum(x.values()) for x in doc_terms) / len(doc_terms)
        df = {w: sum(w in dt for dt in doc_terms) for w in qterms}

        def score(dt: Counter) -> float:
            size = sum(dt.values())
            total = 0.0
            for word in qterms:
                tf = dt[word]
                if not tf:
                    continue
                idf = math.log1p((len(docs) - df[word] + 0.5) / (df[word] + 0.5))
                total += idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * size / avg_len))
            return total

        ranked = sorted(zip(docs, doc_terms),
                        key=lambda pair: (-score(pair[1]), pair[0]["doc_id"]))
        out = [d for d, _ in ranked[:self.top_k]]
        if self.passage_mode == "query_window":
            from .corpus import windows

            # Split hyphenated qualifiers for retrieval only: a query for
            # "syndromic" must surface a sentence saying "non-syndromic" so
            # the answerer can see the negation. Keep the original wording in
            # the selected passage; no semantic conclusion is made here.
            def window_tokens(value: str) -> set[str]:
                split = set(re.findall(r"[a-z0-9]{3,}", value.lower()))
                if self.snapshot_protocol in DOTTED_PROTOCOLS:
                    split.update(w.lower() for w in self.word_pattern.findall(value))
                return split

            window_terms = window_tokens(actual_query)
            for doc in out:
                candidates = windows(doc["text"])
                if candidates:
                    doc["text"] = max(
                        enumerate(candidates),
                        key=lambda item: (len(window_terms & window_tokens(item[1])),
                                          -item[0]))[1]
        for index, doc in enumerate(out):
            doc["rank"] = index
        return out
