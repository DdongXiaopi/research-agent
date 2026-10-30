"""Human/rubric scoring seam for open answers; MCQ pseudo-labels are excluded.

ScholarQABench separates rubric utility and citation attribution. This adapter
accepts provenance-bound independent annotations rather than synthesizing gold
by overlap with the model's own answer. No automatic promotion uses this report.
"""
from ..util import sha256_obj
import random
from .annotation_policy import validate_actor, validate_pair, ORIGINS


def validate_group_partition(partitions: dict[str, list[dict]]) -> dict:
    if set(partitions) != {"dev_optimize", "selection_gate", "final_holdout"}:
        raise ValueError("open evaluation requires three explicit independent partitions")
    owners, task_ids = {}, set()
    for split, tasks in partitions.items():
        if not isinstance(tasks, list) or not tasks:
            raise ValueError("open evaluation partitions cannot be empty")
        for task in tasks:
            identity, group = task.get("task_id"), task.get("group_id")
            if (not isinstance(identity, str) or not identity or identity in task_ids
                    or not isinstance(group, str) or not group):
                raise ValueError("open tasks require unique IDs and independent article/topic groups")
            if group in owners and owners[group] != split:
                raise ValueError("article/topic group crosses an evaluation boundary")
            owners[group] = split
            task_ids.add(identity)
    return {"passed": True, "partition_sha256": sha256_obj(partitions), "groups": len(owners)}


def paired_group_bootstrap(case_rows: list[dict], assessed_cases: dict[str, dict], *,
                           baseline: str, candidate: str, metrics=("rubric", "citation"),
                           B: int = 10000, rng_seed: int = 20261016) -> dict:
    """Paired bootstrap over source groups, not individual questions.

    `case_rows` are private execution metadata and must carry a stable group
    digest. The input intentionally requires exact task/arm pairing; missing or
    unresolved scores fail closed instead of silently shrinking the cohort.
    """
    if (not isinstance(B, int) or isinstance(B, bool) or B < 1 or
            not isinstance(rng_seed, int) or isinstance(rng_seed, bool)):
        raise ValueError("bootstrap count and seed must be valid integers")
    indexed = {}
    for row in case_rows:
        case_id, arm = row.get("case_id"), row.get("arm")
        task, group = row.get("task_sha256"), row.get("group_sha256")
        if not all(isinstance(value, str) and value for value in (case_id, arm, task, group)):
            raise ValueError("group bootstrap requires bound case, arm, task and group identities")
        key = (task, arm)
        if key in indexed:
            raise ValueError("duplicate task/arm in paired group bootstrap")
        indexed[key] = (case_id, group)
    tasks = sorted({task for task, arm in indexed if arm in {baseline, candidate}})
    if not tasks or baseline == candidate:
        raise ValueError("paired bootstrap requires distinct arms and paired tasks")
    for task in tasks:
        if (task, baseline) not in indexed or (task, candidate) not in indexed:
            raise ValueError("baseline/candidate task pairing is incomplete")
        if indexed[(task, baseline)][1] != indexed[(task, candidate)][1]:
            raise ValueError("paired arms disagree on source group")
    groups = sorted({indexed[(task, baseline)][1] for task in tasks})
    output = {}
    for metric in metrics:
        # Reuse the same deterministic group draws for each metric so metric
        # ordering cannot change the reported interval.
        rng = random.Random(rng_seed)
        group_diffs = {group: [] for group in groups}
        for task in tasks:
            case_base, group = indexed[(task, baseline)]
            case_cand, _ = indexed[(task, candidate)]
            try:
                before_row, after_row = assessed_cases[case_base], assessed_cases[case_cand]
                before = (before_row["adjudicated_scores"][metric]
                          if "adjudicated_scores" in before_row else
                          min(before_row["scores_by_reviewer"][metric].values()))
                after = (after_row["adjudicated_scores"][metric]
                         if "adjudicated_scores" in after_row else
                         min(after_row["scores_by_reviewer"][metric].values()))
            except (KeyError, TypeError) as exc:
                raise ValueError("paired group interval requires resolved adjudicated scores") from exc
            if (not isinstance(before, (int, float)) or isinstance(before, bool) or
                    not isinstance(after, (int, float)) or isinstance(after, bool)):
                raise ValueError("paired group interval cannot include missing/non-numeric scores")
            if not 0 <= before <= 1 or not 0 <= after <= 1:
                raise ValueError("paired group scores must lie in [0, 1]")
            group_diffs[group].append(float(after) - float(before))
        group_means = [sum(group_diffs[group]) / len(group_diffs[group]) for group in groups]
        point = sum(group_means) / len(group_means)
        samples = []
        for _ in range(B):
            samples.append(sum(group_means[rng.randrange(len(group_means))]
                               for _ in range(len(group_means))) / len(group_means))
        samples.sort()
        lo = samples[max(0, int(0.025 * B))]
        hi = samples[min(B - 1, int(0.975 * B) - 1)]
        output[metric] = {"delta": round(point, 6), "ci95_low": round(lo, 6),
            "ci95_high": round(hi, 6), "groups": len(groups), "tasks": len(tasks),
            "bootstrap_replicates": B, "rng_seed": rng_seed,
            "method": "paired percentile bootstrap resampling source groups with replacement"}
    return {"baseline": baseline, "candidate": candidate, "metrics": output,
            "unit_of_resampling": "article_or_topic_group"}


def assess_annotations(response: dict, rubric: dict, annotations: list[dict]) -> dict:
    """Two context-separated AI or human reviewers cover all frozen items."""
    response_sha = sha256_obj(response)
    items = rubric.get("items")
    if (rubric.get("schema") != "open-research-rubric-v1" or not isinstance(items, list)
            or not items or rubric.get("response_sha256") != response_sha
            or rubric.get("authorship") not in ORIGINS
            or not str(rubric.get("author") or "").strip()):
        raise ValueError("open scoring requires a frozen AI/human reference rubric")
    validate_actor(rubric["author"], rubric["authorship"], rubric.get("author_provenance"))
    item_ids = [row.get("item_id") for row in items]
    if len(set(item_ids)) != len(item_ids) or any(not isinstance(value, str) or not value for value in item_ids):
        raise ValueError("rubric item identities are invalid")
    rubric_sha = sha256_obj(rubric)
    reviewers, entries, actors = set(), {}, {}
    expected = {(kind, identity) for kind, identity in
        [("rubric", value) for value in item_ids] +
        [("citation", str(i)) for i, _ in enumerate(response.get("citations") or [])]}
    for row in annotations:
        reviewer, kind, identity = row.get("reviewer"), row.get("kind"), row.get("item_id")
        value = row.get("supported")
        if (not isinstance(reviewer, str) or not reviewer.strip() or reviewer == rubric["author"]
                or row.get("source") not in ORIGINS or not isinstance(value, bool)
                or not isinstance(row.get("rationale"), str) or not row["rationale"].strip()
                or row.get("response_sha256") != response_sha or row.get("rubric_sha256") != rubric_sha
                or (kind, identity) not in expected or (reviewer, kind, identity) in entries):
            raise ValueError("annotation is missing independent identity, rationale or exact subject binding")
        reviewers.add(reviewer)
        actor = validate_actor(reviewer, row["source"], row.get("reviewer_provenance"))
        if actor["origin"] == "independent_ai" and actor["rubric_sha256"] != rubric_sha:
            raise ValueError("AI reviewer provenance does not bind the frozen rubric")
        if reviewer in actors and actors[reviewer] != actor:
            raise ValueError("reviewer provenance changed within one assessment")
        actors[reviewer] = actor
        entries[(reviewer, kind, identity)] = value
    if len(reviewers) != 2 or set(entries) != {(reviewer, kind, identity)
            for reviewer in reviewers for kind, identity in expected}:
        raise ValueError("two independent reviewers must cover every rubric and citation item")
    first, second = sorted(reviewers)
    origin = validate_pair(list(actors.values()), rubric["author"], rubric.get("author_provenance"))
    agreement = sum(entries[(first, kind, identity)] == entries[(second, kind, identity)]
                    for kind, identity in expected) / len(expected)
    scores = {}
    for kind in ("rubric", "citation"):
        identities = [identity for item_kind, identity in expected if item_kind == kind]
        scores[kind] = {reviewer: sum(entries[(reviewer, kind, identity)] for identity in identities) / len(identities)
                        if identities else None for reviewer in reviewers}
    return {"schema": "open-research-assessment-v2", "response_sha256": response_sha,
        "rubric_sha256": rubric_sha, "reviewers": sorted(reviewers), "agreement": agreement,
        "scores_by_reviewer": scores, "promotion_authorized": False,
        "disagreements_need_adjudication": agreement < 1,
        "annotation_origin": origin, "reviewer_provenance": actors,
        "interpretation": "independent annotations for this response; no generalization or gain claim"}


def citation_statistics(cases: list[tuple[dict, list[dict], list[dict]]]) -> dict:
    """Summarize already-validated citation judgments with explicit denominators.

    A response without citations has no citation-precision denominator. It is
    counted in coverage instead of erasing judgments on other responses. An
    unresolved judgment on an existing citation still makes precision unknown.
    Claims here are only those explicitly declared by citation records; these
    annotations do not enumerate uncited claims or prove joint-source support.
    """
    total = supported = unresolved = responses_with_citations = 0
    declared_claims = supported_claims = unresolved_claims = 0
    macro_scores = []
    for response, annotations, adjudications in cases:
        citations = response.get("citations") or []
        if not citations:
            continue
        responses_with_citations += 1
        chosen = {(row["kind"], row["item_id"]): row["supported"] for row in adjudications}
        outcomes = []
        claims = {}
        for index, citation in enumerate(citations):
            key = ("citation", str(index))
            votes = [row["supported"] for row in annotations
                     if (row["kind"], row["item_id"]) == key]
            value = chosen[key] if key in chosen else (
                votes[0] if len(votes) == 2 and votes[0] == votes[1] else None)
            outcomes.append(value)
            claim = citation.get("claim")
            if isinstance(claim, str) and claim.strip():
                claims.setdefault(claim.strip(), []).append(value)
        total += len(outcomes)
        supported += sum(value is True for value in outcomes)
        unresolved += sum(value is None for value in outcomes)
        if all(value is not None for value in outcomes):
            macro_scores.append(sum(outcomes) / len(outcomes))
        declared_claims += len(claims)
        supported_claims += sum(any(value is True for value in values) for values in claims.values())
        unresolved_claims += sum(not any(value is True for value in values)
                                 and any(value is None for value in values) for values in claims.values())
    return {
        "n_responses": len(cases), "responses_with_citations": responses_with_citations,
        "zero_citation_responses": len(cases) - responses_with_citations,
        "citation_response_coverage": responses_with_citations / len(cases) if cases else None,
        "citation_count": total, "supported_citation_count": supported,
        "unresolved_citation_count": unresolved,
        "citation_precision_micro": supported / total if total and not unresolved else None,
        "citation_precision_macro": (sum(macro_scores) / len(macro_scores)
                                     if macro_scores and not unresolved else None),
        "declared_cited_claim_count": declared_claims,
        "claims_with_individually_supported_citation": supported_claims,
        "declared_claim_support_rate": (supported_claims / declared_claims
                                        if declared_claims and not unresolved_claims else None),
        "citation_recall": None, "unsupported_claim_rate": None,
        "unavailable_reason": "No exhaustive independent inventory of all answer claims or joint-source entailment labels.",
        "precision_denominator": "emitted citation links; zero-citation responses reported separately",
        "claim_denominator": "distinct citation.claim strings only; excludes unenumerated uncited claims",
        "support_basis": "independent semantic annotations, not source-identity or quote matching",
    }
