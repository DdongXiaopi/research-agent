"""Offline G09 v7 capture-plan completeness checks and summary.

This module deliberately performs no I/O beyond reading caller-supplied JSON
in the CLI wrapper. It does not contact Europe PMC, construct a cohort, author
rubrics, answer tasks, or score responses.
"""
from __future__ import annotations

from .open_research_v7_query_plan import V7_PLAN_SCHEMA
from .open_research_v7_query_plan import V7_MIN_REQUEST_INTERVAL_SECONDS


def summarize_v7_capture_plan(plan: dict) -> dict:
    """Return a compact readiness summary for an already hash-validated plan.

    The query-plan validator remains the authority for reconstructing all plan
    fields from the immutable v5/v6 inputs. This second boundary catches
    malformed or incomplete summaries before a CLI presents a plan as ready.
    """
    if not isinstance(plan, dict) or plan.get("schema") != V7_PLAN_SCHEMA:
        raise ValueError("unsupported G09 v7 capture plan")
    arms = plan.get("arms")
    incomplete = plan.get("incomplete_primary_topic_ids")
    if not isinstance(arms, list) or not arms or not isinstance(incomplete, list) or not incomplete:
        raise ValueError("v7 plan must contain arms for incomplete primary topics")
    arm_ids = [row.get("arm_id") for row in arms if isinstance(row, dict)]
    arm_topics = [row.get("topic_id") for row in arms if isinstance(row, dict)]
    if len(arm_ids) != len(arms) or any(not isinstance(value, str) or not value for value in arm_ids):
        raise ValueError("v7 plan contains a malformed query arm")
    if len(set(arm_ids)) != len(arm_ids):
        raise ValueError("v7 plan contains duplicate arm IDs")
    if not set(arm_topics).issubset(set(incomplete)):
        raise ValueError("v7 query arm references a topic outside the incomplete primary set")
    if set(arm_topics) != set(incomplete):
        raise ValueError("v7 plan leaves an incomplete primary topic without a discovery arm")

    max_new = plan.get("max_new_requests")
    base = plan.get("base_request_budget")
    reserve = plan.get("reserve_request_budget")
    page_caps = [row.get("max_pages") for row in arms]
    if (any(type(value) is not int for value in (max_new, base, reserve)) or
            any(type(value) is not int or value < 1 for value in page_caps)):
        raise ValueError("v7 request budgets must be integers")
    arm_base = sum(page_caps)
    if base + reserve != max_new or arm_base != base or max_new < 1:
        raise ValueError("v7 request budget does not cover every arm exactly")
    interval = plan.get("minimum_request_interval_seconds")
    if (type(interval) not in (int, float) or
            interval < V7_MIN_REQUEST_INTERVAL_SECONDS):
        raise ValueError("v7 request pacing must be at least one second between starts")
    current = plan.get("current_complete_primary_groups")
    required = plan.get("required_complete_primary_groups")
    if type(current) is not int or type(required) is not int:
        raise ValueError("v7 plan group counts must be integers")
    if current >= required:
        raise ValueError("v7 discovery plan is unnecessary after the source-group threshold")
    if required != 48:
        raise ValueError("G09 v7 source-group threshold differs from the frozen requirement")
    if plan.get("identity_exclusion_policy", {}).get("replay_parent_doi_or_pmcid") is not False:
        raise ValueError("v7 plan does not prohibit replaying parent source identities")
    policy = plan.get("selection_policy", {})
    if not isinstance(policy, dict) or "not relaxed" not in policy.get("acceptance_gate", ""):
        raise ValueError("v7 plan does not explicitly preserve the strict source acceptance gate")
    if policy.get("cohort") != "source-discovery plan only; no cohort, rubric, answer, annotation or score":
        raise ValueError("v7 plan scope must remain source discovery only")

    return {
        "schema": "g09-v7-capture-plan-preflight-v1",
        "plan_sha256": plan.get("plan_sha256"),
        "parent_complete_primary_groups": plan["current_complete_primary_groups"],
        "required_complete_primary_groups": plan["required_complete_primary_groups"],
        "incomplete_primary_topic_count": len(incomplete),
        "query_arm_count": len(arms),
        "base_request_budget": base,
        "reserve_request_budget": reserve,
        "max_new_requests": max_new,
        "minimum_request_interval_seconds": interval,
        "parent_cumulative_requests": plan.get("parent_cumulative_request_attempts"),
        "cumulative_request_hard_max": plan.get("cumulative_request_hard_max"),
        "source_acceptance_gate_relaxed": False,
        "parent_identity_replay_allowed": False,
        "cohort_created": False,
        "rubric_or_answer_or_score_created": False,
        "network_calls_made": 0,
        "model_calls_made": 0,
        "capture_authorization_required": True,
    }
