# Vendor and dataset licence audit — 2026-10-10

## Scope and conclusion

This is a source-and-file audit, not legal advice or a blanket redistribution clearance. It corrects the old `THIRD_PARTY.yml` statement that every vendor snapshot lacks a licence. Several snapshots do contain root licence files, package declarations, or dataset-card licence metadata. Those signals apply only to the named repository, package, or dataset; they do not automatically cover nested packages, benchmark components, transitive dependencies, or individual Skills.

No third-party Skill was imported into the M1 runtime as part of this audit. The six self-authored M1 Skills remain the runtime set. The scientific-agent-skills repository contains heterogeneous per-Skill metadata; for example, its `docx` Skill says “Proprietary” and points to terms in `LICENSE.txt`. A repository-level MIT licence is not sufficient to approve an unselected Skill.

2026-10-11 active-snapshot clarification: “six” above describes the historical `baseline-v0` set. The current isolated CliniFact v3 release contains **ten** self-authored Skills from `configs/task_input/registry.v2.yaml` and the listed local bodies. `load_state("task-input-clinifact-v3")` verified all ten nonempty bodies against the immutable release hashes; every registry entry declares `SELF-AUTHORED`. No third-party Skill was imported into that release. This scoped check does not establish redistribution terms for future selected vendor Skills.

## Verified local evidence

| Lock entry / scope | Local evidence | Recorded result | Remaining boundary |
|---|---|---|---|
| pi repository | `vendor/pi-main/pi-main/LICENSE` SHA-256 `0457f5bcec3b3b211605dfb5d1a49042fd638f3686a410fe099c24a25af13c48` | MIT | Root licence does not settle the many bundled `node_modules` dependency licences. Snapshot commit is not recoverable from the zip alone. |
| DSPy repository | Existing lock entry pins `LICENSE` SHA-256 `fc84421fcf968a696680dacf66ad53db769f699f557690127c5ccb3f6358cfdb` | MIT | Package version is pinned; this zip does not itself prove the original Git commit. |
| GEPA repository | Existing lock entry pins `LICENSE` SHA-256 `10c47467a961feb40adf3294fe27dd9cba79d4d1b7cf27173b1c34586d4126c3` | MIT | Source-tree digest pins content, not an upstream Git commit identity. |
| pk-pi-hermes-evolve repository | `vendor/pk-pi-hermes-evolve-main/pk-pi-hermes-evolve-main/LICENSE` SHA-256 `1806c2e126908da6721b16b9e6fc33ad3726b7012ca75675d132e129875da03e` | MIT | Snapshot commit remains unverified. |
| hermes-agent-self-evolution repository | `pyproject.toml` declares MIT; no root LICENSE file in snapshot | MIT declaration recorded, source text absent locally | Declaration is not a vendored licence text; snapshot commit remains unverified. |
| AstaBench repository | `vendor/asta-bench-main/asta-bench-main/LICENSE` SHA-256 `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4`; `NOTICE.txt` is present | Apache-2.0 | Does not clear separately sourced datasets. |
| Scientific Agent Skills repository | `LICENSE.md` SHA-256 `09b02a3c9df3053c55531d503357a9c7cde275970e6c3ceaa1ddf5f0e90b40c1` | MIT at repository level | Individual Skill metadata and referenced terms control each Skill; import set is not yet chosen or audited. |
| PaperQA repository | root `LICENSE` SHA-256 `a10f591e3522fbfe867cfde0b35e0594f6f43a51b013540208a1469c9e0b0f62` | Apache-2.0 at repository root; locked reuse is `src/paperqa/clients` | A nested, unused `packages/paper-qa-pymupdf/LICENSE` is AGPL-3.0; do not generalize the root licence to it. |
| LitQA2 and ProtocolQA datasets | Local Parquet SHA-256s are recorded in their manifests; official dataset cards specify `cc-by-sa-4.0` | CC-BY-SA-4.0 for the respective public datasets | Other LAB-Bench subtasks and source papers may have distinct rights; this row covers only these benchmark files. |
| ScholarQABench aggregate data | Local root `LICENSE` SHA-256 `59127d0cd4e59e4d25a489b9a66828b7fb21407f778da299206af55741c45a9a`; README distinguishes data terms | ODC-BY for aggregate data; MIT for repository code | Constituent datasets retain their own terms. |
| DeepResearch Bench repository | Local `LICENSE` SHA-256 `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4` | Apache-2.0 according to the file and official repo licence | README badge conflicts (MIT); repository code licence does not establish data/evaluation-content terms. Treat the content licence as unresolved. |
| SuppQA snapshot | Only the Parquet file is present in the local path; no accompanying licence file found | UNVERIFIED; not used on the mainline | Do not infer the file's rights from another LAB-Bench dataset without a source-specific record. |

The `docx` Skill under Scientific Agent Skills provides a concrete example of why per-Skill review matters: its frontmatter marks it proprietary and its adjacent `LICENSE.txt` refers to Anthropic service terms. It has not been imported.

## Official source cross-checks

- Pi: <https://github.com/earendil-works/pi/blob/main/LICENSE>
- DSPy: <https://github.com/stanfordnlp/dspy/blob/main/LICENSE>
- GEPA: <https://github.com/gepa-ai/gepa/blob/main/LICENSE>
- pk-pi-hermes-evolve: <https://github.com/kingkillery/pk-pi-hermes-evolve>
- Hermes Agent Self-Evolution: <https://github.com/NousResearch/hermes-agent-self-evolution>
- AstaBench: <https://github.com/allenai/asta-bench>
- Scientific Agent Skills: <https://github.com/K-Dense-AI/scientific-agent-skills>; per-Skill example <https://github.com/K-Dense-AI/scientific-agent-skills/blob/main/skills/docx/SKILL.md>
- PaperQA: <https://github.com/Future-House/paper-qa/blob/main/LICENSE>
- LitQA2 card: <https://huggingface.co/datasets/futurehouse/lab-bench/tree/f0732e66e4ef8aed869fc7009a828502f0128fdb/LitQA2>
- ProtocolQA card: <https://huggingface.co/datasets/futurehouse/lab-bench/tree/bdd4f03dab3487673729411183b461f2ef670660/ProtocolQA>
- LAB-Bench repository declaration: <https://github.com/GeneralReasoning/LAB-Bench>
- ScholarQABench data terms: <https://github.com/AkariAsai/ScholarQABench>
- DeepResearch Bench: <https://github.com/Ayanami0730/deep_research_bench/blob/main/LICENSE>

## Acceptance status and non-goals

`UPSTREAM_LOCK.yml` now records the above evidence and distinctions. Because that lock is included in protocol source fingerprints, any plan made before this metadata update must be considered stale and regenerated before execution. This is a one-time audit update; do not amend the lock for cosmetic wording. Add further changes only when they correct evidence, scope, or a required pin.

This audit does not close the Spec requirement to pin every consumed upstream revision. Several zip snapshots have no `.git` metadata or trusted commit identifier; their local file/source hashes provide content identity but are not commit provenance. It also does not complete the future M2 Skill selection review, the transitive dependency inventory for distribution, or legal review of every evaluation data constituent.

## Installed Pi bridge npm dependency inventory (2026-10-11)

The read-only inventory at `artifacts/provenance/pi-bridge-dependency-license-inventory-20261011.json` (SHA-256 `a9015fbecdd718fe0c06dbbda0ab4a125b075b7e2592d937072f699f9aca33ad`) compares each `tools/pi-bridge/package-lock.json` v3 package entry with its installed `node_modules` package metadata. Of 117 locked dependency packages, 92 are installed with matching versions and lock integrity fields. The 25 absent entries are optional esbuild binaries for other platforms; no required package is absent. A separate `npm ls --all --json --offline` completed with exit code 0 and no reported dependency problems.

All 92 installed packages declare a licence in `package.json`: 43 MIT, 34 Apache-2.0, 12 BSD-3-Clause, and one each ISC, 0BSD and Unlicense. Twelve have no separate licence file at the installed package root, including the two direct Pi packages. Their metadata declarations are evidence of the package's claim, not a completed legal review of redistribution terms or NOTICE obligations. The machine record contains package identity, lock integrity, and local licence-file hashes without application configuration or credentials. This closes the *installed-tree inventory* gap for this Pi bridge snapshot; distribution clearance and other vendored trees remain open.

## Release-coordinate follow-up (2026-10-10)

A read-only official-GitHub check found immutable release coordinates that agree with the version metadata in the vendored snapshots:

| Local snapshot metadata | Official release coordinate | Identity evidence | Limit |
|---|---|---|---|
| Pi `packages/agent/package.json` = `0.87.1` | `v0.87.1`, commit `f07218c4d4bbc12bef056a7058c3dd49dfe41abe` | Official release commit changes the agent package version from 0.87.0 to 0.87.1 | Matching package metadata is not a full local-tree-to-commit comparison. |
| DSPy `dspy/__metadata__.py` = `3.4.0b1` | `3.4.0b1`, commit `ecba33763316d2a4c6c756046a1118ecbff033e7` | Official prerelease tag and commit are published by the upstream repository | The vendored zip has no `.git`; full source-tree equality to this commit remains unverified. |
| GEPA `pyproject.toml` = `0.1.4` | `v0.1.4`, commit `8b0ce6cd99a234f6b74daf37558a2ac0ce18f975` | Official release tag and commit are published by the upstream repository | The local locked source-tree digest identifies the snapshot, but equality to the upstream commit tree remains unverified. |

Primary references: https://github.com/earendil-works/pi/commit/f07218c4d4bbc12bef056a7058c3dd49dfe41abe ; https://github.com/stanfordnlp/dspy/commit/ecba33763316d2a4c6c756046a1118ecbff033e7 ; https://github.com/gepa-ai/gepa/commit/8b0ce6cd99a234f6b74daf37558a2ac0ce18f975 . The workspace vendor directories contain no `.git`. A direct read-only `git ls-remote` attempt could not connect to GitHub from this shell, so no archive-vs-commit tree comparison was possible. This is stronger evidence for the likely release coordinate, not proof of snapshot provenance; the revision-lock criterion remains partial pending a content-to-commit comparison or trusted original archive checksum/provenance.

## Commit archive comparison (2026-10-10)

To resolve the previous comparison gap, downloaded the public GitHub codeload archives by the three immutable commit IDs above and compared local file paths and SHA-256 contents. The downloaded archive digests are DSPy `ec0bc8f69828c0af2e5d4e153e5c52732d73baca34593bb1114e9a2ddcca3577`, GEPA `db2622ae136d4861ac1cb509429423b07871d0770c7b24ab8819bff35eee7c8d`, and Pi `600815c8dc8b6bbe6f989b3dc220deb5f9c2de6ebb9f06c0f2d11447a7cdb277`. Comparison excluded generated Python caches/bytecode; DSPy `.egg-info`; and Pi `node_modules`.

| Snapshot scope | Local vs commit archive | Result |
|---|---:|---|
| DSPy MIPRO optimizer plus seven proposal/generation files already listed in `UPSTREAM_LOCK.yml` | 8 / 8 files byte-identical | The MIPRO code path is verified against commit `ecba33763316d2a4c6c756046a1118ecbff033e7`. |
| Full DSPy repository snapshot | 709 local vs 685 archive files; 104 changed, 25 local-only, 1 archive-only | The whole ZIP is not that exact commit, even though its version metadata and the eight MIPRO files match. |
| GEPA `src/gepa` runtime source | 134 local vs 114 archive files; 25 changed, 20 local-only, 0 archive-only | Local tree SHA-256 `e03a7a8f7b2d188d853e8990fd20c01c5bf522aaeb0bac85704cdc05e8775ab5`; commit tree SHA-256 `598264cf38c4f36048ba88edc05ea2588edcbc4ac7451b8409d71c5f421a48b0`. They differ. |
| Pi `packages/agent/src/harness` | 109 local vs 108 archive files; 2 changed, 1 local-only, 0 archive-only | Local tree SHA-256 `35e6b9f0058394c129943aba205dfd8881d958c86e361a80a70aee6e2da62755`; commit tree SHA-256 `930e5c5cf538a923e629784051e3c360f18bc9174aa3d25f8f0c99ebcb6e5002`. `pico3/session.ts` and `pico3/view.ts` differ; `pico3/legacy-tracker.ts` exists only locally. |

Conclusion: the official release coordinates are real, but they are not full-snapshot provenance for the local GEPA or Pi sources; the complete DSPy snapshot also contains unrelated drift. Do not add these three commit IDs to `UPSTREAM_LOCK.yml` as if they identified the local snapshots. The current local source/content hashes remain useful exact identities, and the eight locked DSPy MIPRO files are independently commit-matched, but Spec §1.6 remains **partial** for full upstream revision provenance. Closing that gap requires either a reviewed re-vendoring from a chosen exact commit with compatibility tests, or trusted source provenance for the current local snapshots. No vendor source, lock file, protocol source, or runtime package was changed in this audit.
