"""Deterministic task material builder for open research evaluation.

The default builder requires explicit CC BY Europe PMC records.  Personal
project captures may opt into a provenance-only license policy, which keeps
the source license metadata for audit but does not make it a cohort gate.
The builder creates task-only inputs; answer rubrics must still be authored
separately before any assistant response is generated.
"""
from __future__ import annotations

import hashlib
from collections import Counter

from ..runtime.research_assistant import request_from_dict
from ..util import sha256_obj, sha256_text

TASK_TEMPLATES = (
    ("methods", "Using only the supplied article abstract, describe the study design and the main method or comparison. If a detail is not reported, say so."),
    ("results", "Using only the supplied article abstract, report the main result and any numerical estimate or uncertainty stated. Do not infer causality beyond the reported design."),
    ("population", "Using only the supplied article abstract, identify the studied population or material and the outcome measured. State when a requested detail is not reported."),
    ("limitations", "Using only the supplied article abstract, report the authors' stated limitations or scope caveats. If none are stated in the abstract, say that the abstract does not specify them."),
)
SINGLE_PAPER_TASK_TYPES = tuple(kind for kind, _ in TASK_TEMPLATES)
MULTI_PAPER_TASK_TYPES = ("synthesis", "evidence_insufficient")
DIAGNOSTIC_TASK_TYPES = ("conflict",)
TYPED_TASK_TYPES = SINGLE_PAPER_TASK_TYPES + MULTI_PAPER_TASK_TYPES
PARTITIONS = ("dev_optimize", "selection_gate", "final_holdout")
HISTORICAL_DOI_PREFIX_EXCLUSIONS = ("10.1371/",)
IDENTITY_ONLY_EXPOSURE_NOTE = (
    "Identity-only exposure of old G09 final DOI/URL values: "
    "10.1371/journal.pone.0302385, 10.1371/journal.pone.0297184, "
    "10.1371/journal.pone.0000332; abstract, rubric, response and scores were not read. "
    "Exclude the full PLOS DOI family 10.1371/ to avoid article/source-family overlap."
)
LICENSE_POLICIES = ("cc_by_only", "personal_project_provenance_only")


def _record_identity(record: dict) -> str:
    doi = record.get("doi")
    if not isinstance(doi, str) or not doi.strip() or "/" not in doi:
        raise ValueError("Europe PMC record needs a DOI")
    return doi.strip().casefold()


def validate_source_records(records: list[dict], *, excluded_dois=(),
        license_policy: str = "cc_by_only") -> dict:
    """Validate real, source-licensed records and return a deterministic source summary."""
    if not isinstance(records, list) or not records:
        raise ValueError("at least one Europe PMC article record is required")
    if license_policy not in LICENSE_POLICIES:
        raise ValueError("unsupported source license policy")
    excluded = {str(value).strip().casefold() for value in excluded_dois}
    ids, journals = set(), set()
    normalized = []
    for row in records:
        if not isinstance(row, dict):
            raise ValueError("article records must be objects")
        doi = _record_identity(row)
        if doi in excluded:
            raise ValueError("article DOI appears in the explicit historical exclusion list")
        if doi in ids:
            raise ValueError("duplicate article DOI")
        if (license_policy == "cc_by_only" and
                str(row.get("license", "")).strip().casefold() != "cc by"):
            raise ValueError("only records explicitly marked CC BY are eligible")
        pmcid = row.get("pmcid")
        title, abstract = row.get("title"), row.get("abstractText")
        journal = ((row.get("journalInfo") or {}).get("journal") or {}).get("title")
        year = row.get("pubYear") or ((row.get("journalInfo") or {}).get("journal") or {}).get("yearOfPublication")
        if (not isinstance(pmcid, str) or not pmcid.startswith("PMC") or
                not isinstance(title, str) or len(title.strip()) < 8 or
                not isinstance(abstract, str) or len(abstract.strip()) < 400 or
                not isinstance(journal, str) or not journal.strip() or
                not str(year).isdigit()):
            raise ValueError("article record lacks PMCID, title, abstract, journal or publication year")
        ids.add(doi)
        journals.add(journal.strip())
        normalized.append({"doi": doi, "pmcid": pmcid, "title": title.strip(),
            "abstract": abstract.strip(), "journal": journal.strip(), "year": int(year),
            "pmid": str(row.get("pmid") or "")})
    return {"records": normalized, "record_count": len(normalized),
            "unique_journals": len(journals), "journal_titles": sorted(journals),
            "license_policy": license_policy,
            "source_records_sha256": sha256_obj(normalized)}


def build_unannotated_cohort(records: list[dict], *, revision: str,
        source_capture_at: str, split_seed: str, excluded_dois=(),
        min_records: int = 24, min_journals: int = 8,
        max_per_journal: int = 4, min_tasks_per_type_per_split: int = 1) -> dict:
    """Build a reproducible 3-way, group-disjoint task-only dataset.

    Input records must be a frozen API capture. Selection is deterministic;
    article groups are partitioned as a whole and each contributes two distinct
    research tasks. The returned final split is evaluator-only input and must
    not be opened by development/selection workflows.
    """
    if (not isinstance(revision, str) or not revision.strip() or
            not isinstance(source_capture_at, str) or not source_capture_at.strip() or
            not isinstance(split_seed, str) or not split_seed.strip()):
        raise ValueError("cohort revision, capture time and split seed are required")
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 1
           for v in (min_records, min_journals, max_per_journal)):
        raise ValueError("cohort size and journal caps must be positive integers")
    if (isinstance(min_tasks_per_type_per_split, bool) or
            not isinstance(min_tasks_per_type_per_split, int) or min_tasks_per_type_per_split < 1):
        raise ValueError("each split needs a positive minimum per task type")
    excluded = {str(value).strip().casefold() for value in excluded_dois}
    family_excluded = [row for row in records if isinstance(row, dict) and
        any(str(row.get("doi", "")).strip().casefold().startswith(prefix)
            for prefix in HISTORICAL_DOI_PREFIX_EXCLUSIONS)]
    eligible_records = [row for row in records if row not in family_excluded]
    validated = validate_source_records(eligible_records, excluded_dois=excluded)
    # Stable ordering avoids privileging API ranking or request order.
    candidates = sorted(validated["records"],
        key=lambda row: hashlib.sha256((split_seed + "\0" + row["doi"]).encode()).hexdigest())
    selected, by_journal = [], Counter()
    for row in candidates:
        if by_journal[row["journal"]] >= max_per_journal:
            continue
        selected.append(row)
        by_journal[row["journal"]] += 1
        if len(selected) >= min_records and len(by_journal) >= min_journals:
            break
    if len(selected) < min_records or len(by_journal) < min_journals:
        raise ValueError("licensed source pool cannot meet the frozen group/journal diversity minimum")

    # Assign whole article groups by hash rank, not independent modulo draws,
    # so a valid small cohort cannot accidentally lose one partition.
    ranked = sorted(selected, key=lambda row: hashlib.sha256(
        (split_seed + "\0" + row["doi"]).encode()).hexdigest())
    n_dev = int(len(ranked) * 0.6)
    n_selection = int(len(ranked) * 0.2)
    split_by_doi = {row["doi"]: ("dev_optimize" if index < n_dev else
        "selection_gate" if index < n_dev + n_selection else "final_holdout")
        for index, row in enumerate(ranked)}

    partitions = {name: [] for name in PARTITIONS}
    group_owners = {}
    tasks_by_type = Counter()
    split_group_order = {split: sorted((row for row in selected if split_by_doi[row["doi"]] == split),
        key=lambda row: hashlib.sha256((row["doi"] + "\0task-order").encode()).hexdigest())
        for split in PARTITIONS}
    group_task_types = {}
    for split, split_rows in split_group_order.items():
        for group_index, article in enumerate(split_rows):
            pair_start = (group_index * 2) % len(TASK_TEMPLATES)
            group_task_types[article["doi"]] = [TASK_TEMPLATES[pair_start][0],
                TASK_TEMPLATES[(pair_start + 1) % len(TASK_TEMPLATES)][0]]

    for article in selected:
        doi, group_id = article["doi"], "doi:" + article["doi"]
        split = split_by_doi[doi]
        group_owners[group_id] = split
        source_id = "pmc-" + article["pmcid"].casefold().replace("pmc", "")
        source = {"source_id": source_id, "text": article["abstract"],
            "url": "https://doi.org/" + doi,
            "locator": f"{article['title']} — {article['journal']} ({article['year']}); Europe PMC {article['pmcid']}",
            "retrieved_at": source_capture_at}
        for slot, task_type in enumerate(group_task_types[doi]):
            template = next(question for kind, question in TASK_TEMPLATES if kind == task_type)
            task_id = "g09-" + sha256_text(doi)[:12] + "-" + task_type + "-" + str(slot + 1)
            request = {"task_id": task_id, "task_type": "open_research",
                "project_id": "g09-open-research", "session_id": "group-" + sha256_text(group_id)[:16],
                "query": template, "evidence": [source],
                "budget": {"max_evidence_chars": 12000, "max_answer_chars": 5000,
                           "max_completion_tokens": 1000, "max_physical_requests": 1}}
            request_from_dict(request)
            partitions[split].append({"request": request, "group_id": group_id})
            tasks_by_type[task_type] += 1
    if set(group_owners.values()) != set(PARTITIONS) or min(Counter(group_owners.values()).values()) < 3:
        raise ValueError("deterministic split did not yield at least three article groups per partition")
    if any(len(rows) < 6 for rows in partitions.values()):
        raise ValueError("each benchmark partition needs at least six tasks")
    coverage = {split: Counter(case["request"]["query"] for case in rows)
        for split, rows in partitions.items()}
    type_queries = {question: kind for kind, question in TASK_TEMPLATES}
    coverage_by_type = {split: Counter(type_queries[query] for query in counts for _ in range(counts[query]))
        for split, counts in coverage.items()}
    if any(coverage_by_type[split][kind] < min_tasks_per_type_per_split
           for split in PARTITIONS for kind, _ in TASK_TEMPLATES):
        raise ValueError("a split does not meet minimum task-type coverage")
    source = ("Europe PMC REST core records; individual article license verified as CC BY; "
        "identity-only old G09 final exposure recorded; full PLOS DOI family 10.1371/ excluded")
    provenance = {"source": source, "revision": revision,
        "license": "CC BY (per-record Europe PMC license field)",
        "reference_origin": "AI task-first rubric author pending"}
    result = {"schema": "open-research-unannotated-dataset-v1", "provenance": provenance,
        "partitions": partitions}
    report = {"schema": "g09-open-cohort-build-v1", "source_records_sha256": validated["source_records_sha256"],
        "cohort_input_sha256": sha256_obj(result), "revision": revision,
        "source_capture_at": source_capture_at, "split_seed": split_seed,
        "records": len(selected), "journal_count": len(by_journal),
        "source_records_excluded_by_family": len(family_excluded),
        "historical_excluded_doi_prefixes": list(HISTORICAL_DOI_PREFIX_EXCLUSIONS),
        "historical_identity_only_exposure": IDENTITY_ONLY_EXPOSURE_NOTE,
        "groups_by_split": dict(Counter(group_owners.values())),
        "tasks_by_split": {split: len(rows) for split, rows in partitions.items()},
        "tasks_by_type": dict(sorted(tasks_by_type.items())),
        "tasks_by_type_per_split": {split: dict(coverage_by_type[split]) for split in PARTITIONS},
        "min_tasks_per_type_per_split": min_tasks_per_type_per_split,
        "final_holdout_read_during_build": False,
        "rubrics_authored": False, "assistant_answers_generated": False,
        "dataset_sha256": sha256_obj(result)}
    report["report_sha256"] = sha256_obj({key: value for key, value in report.items() if key != "report_sha256"})
    return {"dataset": result, "report": report}


def build_typed_unannotated_cohort(records: list[dict], topic_groups: list[dict], *,
        revision: str, source_capture_at: str, split_seed: str,
        min_selection_groups: int = 12, min_groups_per_split: int = 12,
        min_tasks_per_type_per_split: int = 1,
        license_policy: str = "cc_by_only") -> dict:
    """Build a group-disjoint cohort covering single- and multi-paper tasks.

    Each topic group contains at least two independently licensed articles. It
    generates four focal-paper tasks, cross-paper synthesis, and a task whose
    requested information is absent from the supplied abstracts. If verified
    opposing passages are present, it also emits a descriptive-only conflict
    diagnostic; conflict availability is never a cohort acceptance gate.
    """
    if any(not isinstance(value, str) or not value.strip()
           for value in (revision, source_capture_at, split_seed)):
        raise ValueError("cohort revision, capture time and split seed are required")
    for name, value in (("selection groups", min_selection_groups),
                        ("groups per split", min_groups_per_split),
                        ("per-type task coverage", min_tasks_per_type_per_split)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"minimum {name} must be a positive integer")
    if not isinstance(topic_groups, list) or not topic_groups:
        raise ValueError("typed cohort requires explicit topic-group specifications")
    eligible_records = [row for row in records if isinstance(row, dict) and
        not any(str(row.get("doi", "")).strip().casefold().startswith(prefix)
                for prefix in HISTORICAL_DOI_PREFIX_EXCLUSIONS)]
    validated = validate_source_records(eligible_records, license_policy=license_policy)
    by_doi = {row["doi"]: row for row in validated["records"]}
    group_ids, article_owners = set(), {}
    normalized_groups = []
    for spec in topic_groups:
        required_keys = {"group_id", "topic", "source_dois", "focal_doi", "insufficient_target"}
        conflict_keys = {"conflict_question", "conflict_evidence"}
        if not isinstance(spec, dict) or not required_keys <= set(spec) or set(spec) - required_keys - conflict_keys:
            raise ValueError("topic group has missing/unsupported identity, source or task metadata")
        if bool(conflict_keys & set(spec)) and not conflict_keys <= set(spec):
            raise ValueError("conflict question and passages must be provided together")
        group_id = spec["group_id"]
        topic = spec["topic"]
        dois = spec["source_dois"]
        if (not isinstance(group_id, str) or not group_id.strip() or group_id in group_ids or
                any(char.isspace() for char in group_id) or
                not isinstance(topic, str) or not topic.strip() or
                not isinstance(dois, list) or len(dois) < 2 or len(dois) > 4 or
                any(not isinstance(doi, str) for doi in dois) or len({doi.casefold() for doi in dois}) != len(dois)):
            raise ValueError("topic group identity/topic/source list is invalid")
        normalized_dois = [doi.strip().casefold() for doi in dois]
        if any(doi not in by_doi for doi in normalized_dois):
            raise ValueError("topic group references a missing or excluded DOI")
        if any(doi in article_owners for doi in normalized_dois):
            raise ValueError("source article may belong to only one indivisible topic group")
        focal_doi = str(spec["focal_doi"]).strip().casefold()
        if focal_doi not in normalized_dois:
            raise ValueError("focal article must be a member of its topic group")
        conflict_question = None
        conflict_evidence = None
        if conflict_keys <= set(spec):
            conflict_question = spec["conflict_question"]
            conflict_evidence = spec["conflict_evidence"]
            if (not isinstance(conflict_question, str) or len(conflict_question.strip()) < 12 or
                    not isinstance(conflict_evidence, list) or len(conflict_evidence) < 2):
                raise ValueError("conflict task needs a question and at least two opposing passages")
            stances, conflict_dois = set(), set()
            for passage in conflict_evidence:
                if not isinstance(passage, dict) or set(passage) != {"doi", "stance", "quote"}:
                    raise ValueError("conflict passages need DOI, stance and exact source quote")
                doi = str(passage["doi"]).strip().casefold()
                if (doi not in normalized_dois or not isinstance(passage["stance"], str) or
                        passage["stance"] not in {"supports", "contradicts"} or
                        not isinstance(passage["quote"], str) or not passage["quote"].strip() or
                        passage["quote"] not in by_doi[doi]["abstract"]):
                    raise ValueError("conflict evidence must be exact text from distinct group sources")
                stances.add(passage["stance"])
                conflict_dois.add(doi)
            if len(stances) != 2 or len(conflict_dois) < 2:
                raise ValueError("conflict evidence must include both opposing stances from distinct papers")
        insufficient_target = spec["insufficient_target"]
        if (not isinstance(insufficient_target, str) or len(insufficient_target.strip()) < 8 or
                any(insufficient_target.strip().casefold() in by_doi[doi]["abstract"].casefold()
                    for doi in normalized_dois)):
            raise ValueError("evidence-insufficient target must be absent verbatim from every group abstract")
        group_ids.add(group_id)
        for doi in normalized_dois:
            article_owners[doi] = group_id
        normalized_groups.append({**spec, "group_id": group_id.strip(), "topic": topic.strip(),
            "source_dois": normalized_dois, "focal_doi": focal_doi,
            "conflict_question": conflict_question.strip() if conflict_question else None,
            "conflict_evidence": conflict_evidence,
            "insufficient_target": insufficient_target.strip()})
    group_count = len(normalized_groups)
    if group_count < 4 * min_groups_per_split:
        raise ValueError("typed cohort has too few topic groups for frozen split minima")
    n_dev = int(group_count * 0.5)
    n_selection = int(group_count * 0.25)
    target_groups = {"dev_optimize": n_dev, "selection_gate": n_selection,
                     "final_holdout": group_count - n_dev - n_selection}
    # Split every topic group with one deterministic hash ordering. Conflict
    # material is sparse and exploratory, so it must not control partitioning.
    ranked_groups = sorted(normalized_groups, key=lambda group: hashlib.sha256(
        (split_seed + "\0group\0" + group["group_id"]).encode()).hexdigest())
    split_by_group = {}
    offset = 0
    for split in PARTITIONS:
        count = target_groups[split]
        for group in ranked_groups[offset:offset + count]:
            split_by_group[group["group_id"]] = split
        offset += count
    ranked = sorted(normalized_groups, key=lambda group: hashlib.sha256(
        (split_seed + "\0task-order\0" + group["group_id"]).encode()).hexdigest())
    groups_by_split = Counter(split_by_group.values())
    conflict_by_split_counts = Counter(split_by_group[group["group_id"]]
        for group in normalized_groups if group["conflict_evidence"])
    if (groups_by_split.get("selection_gate", 0) < min_selection_groups or
            any(groups_by_split.get(split, 0) < min_groups_per_split for split in PARTITIONS)):
        raise ValueError("selection/final split topic-group count is below frozen statistical minimum")
    partitions = {split: [] for split in PARTITIONS}
    tasks_by_split_type = {split: Counter() for split in PARTITIONS}
    group_sha_by_id = {}
    task_order = (("methods",), ("results",), ("population",), ("limitations",),
                  ("synthesis",), ("conflict",), ("evidence_insufficient",))
    for group in ranked:
        group_id = group["group_id"]
        split = split_by_group[group_id]
        group_sha_by_id[group_id] = sha256_text(group_id)
        articles = [by_doi[doi] for doi in group["source_dois"]]
        source_by_doi = {}
        for article in articles:
            source_id = "pmc-" + article["pmcid"].casefold().replace("pmc", "")
            source_by_doi[article["doi"]] = {"source_id": source_id, "text": article["abstract"],
                "url": "https://doi.org/" + article["doi"],
                "locator": f"{article['title']} — {article['journal']} ({article['year']}); Europe PMC {article['pmcid']}",
                "retrieved_at": source_capture_at}
        focal = by_doi[group["focal_doi"]]
        template_by_type = dict(TASK_TEMPLATES)
        query_by_type = {kind: template_by_type[kind].replace("the supplied article abstract", "the focal article abstract")
            .replace("the studied population or material", "the focal article's studied population or material")
            .replace("the main result", "the focal article's main result")
            .replace("the authors' stated limitations", "the focal article authors' stated limitations")
            .replace("study design and the main method or comparison", "focal study design and main method or comparison")
            for kind in SINGLE_PAPER_TASK_TYPES}
        query_by_type.update({
            "synthesis": (f"Compare the findings across the supplied abstracts about {group['topic']}. "
                "Identify agreement and differences, compare study designs/populations/outcomes where reported, "
                "and preserve uncertainty without inferring causality."),
            "evidence_insufficient": (f"Using only the supplied abstracts, answer: {group['insufficient_target']}? "
                "If that information is not explicitly reported, state that the evidence is insufficient and do not infer it."),
        })
        if group["conflict_evidence"]:
            query_by_type["conflict"] = (f"The supplied abstracts report different findings relevant to: {group['conflict_question']} "
                "Summarize each paper's position with source-specific citations, assess whether design, population, "
                "or outcome differences could explain the disagreement, and do not invent a resolution.")
        group_task_types = SINGLE_PAPER_TASK_TYPES + ("synthesis",) + (
            ("conflict",) if group["conflict_evidence"] else ()) + ("evidence_insufficient",)
        for task_type in group_task_types:
            source_rows = ([source_by_doi[focal["doi"]]] if task_type in SINGLE_PAPER_TASK_TYPES else
                           [source_by_doi[doi] for doi in group["source_dois"]])
            task_id = "g09-" + task_type + "-" + sha256_text(group_id + "\0" + task_type)[:16]
            request = {"task_id": task_id, "task_type": "open_research",
                "project_id": "g09-open-research", "session_id": "group-" + sha256_text(group_id)[:16],
                "query": query_by_type[task_type], "evidence": source_rows,
                "budget": {"max_evidence_chars": 24000, "max_answer_chars": 5000,
                # Open answers may contain several claims plus verbatim evidence.
                # Keep the frozen budget below the runtime ceiling while avoiding
                # deterministic truncation from DeepSeek Flash on multi-source tasks.
                "max_completion_tokens": 4096, "max_physical_requests": 1}}
            request_from_dict(request)
            partitions[split].append({"request": request, "group_id": group_id,
                "evaluation_role": "diagnostic_only" if task_type == "conflict" else "primary"})
            tasks_by_split_type[split][task_type] += 1
    coverage_failure = [
        (split, task_type) for split in PARTITIONS for task_type in TYPED_TASK_TYPES
        if tasks_by_split_type[split][task_type] < min_tasks_per_type_per_split]
    if coverage_failure:
        raise ValueError("a split does not meet minimum per-task-family coverage")
    license_text = ("CC BY (per-record Europe PMC license field)" if license_policy == "cc_by_only"
        else "license metadata retained for provenance only; personal project policy")
    source_text = ("Europe PMC REST core; per-record CC BY; ScholarQABench multi-paper/citation separation adapted; PLOS DOI family excluded"
        if license_policy == "cc_by_only" else
        "Europe PMC REST core; license metadata retained for provenance only under personal project policy; ScholarQABench multi-paper/citation separation adapted; PLOS DOI family excluded")
    result = {"schema": "open-research-unannotated-dataset-v1",
        "provenance": {"source": source_text,
            "revision": revision, "license": license_text,
            "license_policy": license_policy,
            "reference_origin": "AI task-first rubric author pending"},
        "partitions": partitions}
    report = {"schema": "g09-open-typed-cohort-build-v1", "source_records_sha256": validated["source_records_sha256"],
        "cohort_input_sha256": sha256_obj(result), "revision": revision,
        "source_capture_at": source_capture_at, "split_seed": split_seed,
        "records": len(article_owners), "topic_groups": group_count,
        "groups_by_split": dict(groups_by_split),
        "tasks_by_split": {split: len(rows) for split, rows in partitions.items()},
        "task_types": list(TYPED_TASK_TYPES),
        "tasks_by_type_per_split": {split: dict(tasks_by_split_type[split]) for split in PARTITIONS},
        "min_selection_groups": min_selection_groups,
        "min_groups_per_split": min_groups_per_split,
        "license_policy": license_policy,
        "diagnostic_task_types": list(DIAGNOSTIC_TASK_TYPES),
        "conflict_evaluation": "descriptive_diagnostic_only",
        "conflict_subgroup_inference_permitted": False,
        "conflict_groups_by_split": dict(conflict_by_split_counts),
        "min_tasks_per_type_per_split": min_tasks_per_type_per_split,
        "group_source_sha256": {group["group_id"]: sha256_text("\0".join(sorted(group["source_dois"])))
            for group in sorted(normalized_groups, key=lambda row: row["group_id"])},
        "final_holdout_read_during_build": False,
        "rubrics_authored": False, "assistant_answers_generated": False,
        "dataset_sha256": sha256_obj(result)}
    report["report_sha256"] = sha256_obj({key: value for key, value in report.items() if key != "report_sha256"})
    return {"dataset": result, "report": report}
