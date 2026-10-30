"""Authoritative tool names available to the task runtime.

Configuration, candidate gates, and execution must share one allowlist so a
mistyped or out-of-scope tool cannot survive until model evaluation.
"""

CONTROLLED_TOOLS = frozenset({
    "workspace.read",
    "artifact.write",
    "academic.search",
    "fulltext.resolve",
    "citation.verify",
    "model.call.task",
})
