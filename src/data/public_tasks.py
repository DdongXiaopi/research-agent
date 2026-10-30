"""Prepare user-supplied MCQ tasks without historical research assets."""
from __future__ import annotations

import re
import unicodedata
from .split import DEFAULT_RATIOS, SPLIT_ORDER, assign_split
from ..util import now_iso, sha256_obj


def prepare(dataset, rows, *, source, license, salt="public-task-input-v1"):
    if not isinstance(dataset, str) or not re.fullmatch(r"public_[a-z0-9_]{1,60}", dataset):
        raise ValueError("public dataset names must start with public_ and use lowercase letters/digits/_")
    if not isinstance(source, str) or not source.strip() or not isinstance(license, str) or not license.strip():
        raise ValueError("public tasks require explicit source and license provenance")
    if not isinstance(rows, list) or not rows:
        raise ValueError("public task input must contain tasks")
    if not isinstance(salt, str) or not salt.strip():
        raise ValueError("public task split salt must be nonempty text")
    prepared, identifiers, owners, parents = [], set(), {}, list(range(len(rows)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError("each public task must be an object")
        allowed = {"task_id", "question", "options", "correct_index", "evidence", "sources", "domain", "group_id"}
        if set(row) - allowed:
            raise ValueError("unsupported public task fields; split is assigned by preparation")
        for key in ("task_id", "question", "evidence"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"public task requires nonempty {key}")
        task_id = row["task_id"].strip()
        if task_id in identifiers:
            raise ValueError("duplicate public task IDs")
        identifiers.add(task_id)
        options, gold = row.get("options"), row.get("correct_index")
        if (not isinstance(options, list) or not 2 <= len(options) <= 30
                or any(not isinstance(x, str) or not x.strip() for x in options)
                or len({x.strip() for x in options}) != len(options) or isinstance(gold, bool)
                or not isinstance(gold, int) or not 0 <= gold < len(options)):
            raise ValueError("public MCQ requires unique options and an in-range integer correct_index")
        sources = row.get("sources", [])
        if not isinstance(sources, list) or any(not isinstance(x, str) or not x.strip() for x in sources):
            raise ValueError("public sources must be nonempty strings")
        sources = [value.strip() for value in sources]
        if "group_id" in row and (not isinstance(row["group_id"], str) or not row["group_id"].strip()):
            raise ValueError("group_id must be nonempty text")
        domain = row.get("domain", "biomedical")
        if not isinstance(domain, str) or not domain.strip():
            raise ValueError("public task domain must be nonempty text")
        normalized_evidence = " ".join(unicodedata.normalize("NFKC", row["evidence"]).split())
        keys = ["text:" + sha256_obj(normalized_evidence)] + ["source:" + x for x in sources]
        if row.get("group_id"):
            keys.append("group:" + row["group_id"].strip())
        for key in keys:
            if key in owners:
                parents[find(i)] = find(owners[key])
            owners[key] = i
        prepared.append({"task_id": task_id, "question": row["question"], "options": [option.strip() for option in options],
                         "correct_index": gold, "evidence": row["evidence"], "context": "",
                         "sources": sources, "dataset": dataset, "task_type": "mcq",
                         "domain": domain.strip()})
    members = {}
    for i, task in enumerate(prepared):
        members.setdefault(find(i), []).append(task["task_id"])
    split_ids = {name: [] for name in SPLIT_ORDER}
    for i, task in enumerate(prepared):
        group = sha256_obj(sorted(members[find(i)]))
        task["group_id"] = group
        task["split"] = assign_split(dataset, group, salt)
        split_ids[task["split"]].append(task["task_id"])
    if not all(split_ids.values()):
        raise ValueError("not enough independent source groups to populate all three splits")
    manifest = {"dataset": dataset, "schema": "public-task-input-dataset-v1",
                "source_file": source, "source_sha256": sha256_obj(rows), "license": license,
                "n_questions": len(rows), "n_source_groups": len(members), "salt": salt,
                "ratios": DEFAULT_RATIOS, "task_type": "mcq", "created_at": now_iso(),
                "split_algorithm": "source-connected-component / sha256(salt|dataset|group)[:8] % 100",
                "splits": {name: {"n": len(ids), "ids": sorted(ids), "ids_sha256": sha256_obj(sorted(ids))}
                           for name, ids in split_ids.items()}}
    return manifest, [x for x in prepared if x["split"] != "final_holdout"], [
        x for x in prepared if x["split"] == "final_holdout"]
