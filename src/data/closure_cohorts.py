"""Independent, source-native binary confirmation cohorts for a fixed release.

These are confirmation sets, not new optimizer train/validation sets. No labels
or gold evidence selection are used to construct the QASPER model input.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from ..util import sha256_file, sha256_obj

BIOASQ_DATASET = "bioasq13_independent_confirmation_20261019"
QASPER_DATASET = "qasper_binary_transfer_20261019"
SPLITS = ("dev_optimize", "selection_gate", "final_holdout")


def norm(text):
    return " ".join(str(text or "").split()).casefold()


def identity(text):
    return hashlib.sha256(norm(text).encode()).hexdigest()


def source_key(url):
    # BioASQ mirrors use http/https and optional trailing slashes for a PMID.
    match = re.search(r"(?:pubmed/|pubmed\.ncbi\.nlm\.nih\.gov/)(\d+)", url)
    return "pmid:" + match[1] if match else url.strip().rstrip('/').casefold()


def bioasq_rows(raw, historical_ids=(), historical_texts=(), historical_documents=()):
    old_ids, old_texts = set(historical_ids), set(historical_texts)
    old_documents = {source_key(u) for r in raw if r['asq_challenge'] < 13
                     for u in r['documents']}
    old_documents.update(historical_documents)
    out, excluded = [], Counter()
    for row in raw:
        if row['asq_challenge'] != 13 or row['split_13b'] != 'test':
            continue
        qid = str(row['id'])
        docs = {source_key(u) for u in row['documents']}
        snippets = list(dict.fromkeys(s['text'] for s in row['snippets'] if s.get('text')))
        if qid in old_ids or identity(row['question']) in old_texts:
            excluded['historical_question'] += 1
            continue
        if docs & old_documents or any(identity(s) in old_texts for s in snippets):
            excluded['historical_article_or_snippet'] += 1
            continue
        if row.get('answer') not in ([['yes']], [['no']]) or not docs or not snippets:
            excluded['invalid_native_reference_or_material'] += 1
            continue
        out.append(make_row(BIOASQ_DATASET, qid, row['question'],
                            '\n\n'.join(snippets), row['answer'][0][0],
                            sorted(docs), 'biomedical',
                            {'upstream_challenge': 13, 'input_scope': 'source-provided gold snippets; oracle evidence setting'}))
    return out, dict(excluded)


def qasper_rows(papers, historical_texts=(), historical_paper_ids=(), max_chars=40000):
    old_texts, old_ids = set(historical_texts), set(historical_paper_ids)
    out, excluded = [], Counter()
    for pid, paper in sorted(papers.items()):
        if pid in old_ids or identity(paper['title']) in old_texts:
            excluded['historical_paper'] += len(paper['qas'])
            continue
        paragraphs = [p for s in paper['full_text'] for p in s['paragraphs']]
        if any(len(p) >= 100 and identity(p) in old_texts for p in paragraphs):
            excluded['historical_paragraph'] += len(paper['qas'])
            continue
        material = paper['title'] + '\n\n' + paper['abstract'] + '\n\n' + '\n\n'.join(
            (s.get('section_name') or '') + '\n' + '\n'.join(s['paragraphs'])
            for s in paper['full_text'])
        if not paragraphs or len(material) > max_chars:
            excluded['whole_paper_exceeds_input_bound'] += len(paper['qas'])
            continue
        eligible = []
        for q in paper['qas']:
            answers = [a['answer'] for a in q['answers']]
            # A majority vote must not manufacture a binary gold label from
            # annotators that disagree on answer type, answerability or label.
            if not answers or any(type(a.get('yes_no')) is not bool or a.get('unanswerable')
                    or a.get('extractive_spans') or a.get('free_form_answer') for a in answers):
                excluded['non_unanimous_binary_type'] += 1
                continue
            if len({a['yes_no'] for a in answers}) != 1:
                excluded['conflicting_native_answers'] += 1
                continue
            # The task is fixed-text QA, not figure/table understanding. Check
            # that native supporting evidence is present without using it to
            # select or reorder the model's full-paper input.
            if any(not a.get('evidence') or any(not isinstance(e, str) or not norm(e)
                    or norm(e) not in norm(material) for e in a['evidence']) for a in answers):
                excluded['native_support_not_in_supplied_body'] += 1
                continue
            if identity(q['question']) in old_texts:
                excluded['historical_question'] += 1
                continue
            eligible.append(make_row(QASPER_DATASET, q['question_id'], q['question'], material,
                             'yes' if answers[0]['yes_no'] else 'no', ['arxiv:' + pid],
                             'computer_science_nlp', {'upstream_paper_id': pid,
                              'native_answer_count': len(answers),
                              'native_support_verified_in_full_body': True,
                              'input_scope': 'whole text, no gold evidence filtering; figures/tables excluded'}))
        # One question per paper avoids overweighting papers. Choice is fixed
        # by question ID hash, not difficulty, label, or observed performance.
        if eligible:
            out.append(min(eligible, key=lambda r: identity(r['task_id'])))
    return out, dict(excluded)


def make_row(dataset, qid, question, evidence, label, sources, domain, metadata):
    return {'dataset': dataset, 'task_id': dataset + '-' + qid,
            'task_type': 'mcq', 'domain': domain, 'question': question,
            'context': '', 'evidence': evidence, 'options': ['yes', 'no'],
            'correct_index': ['yes', 'no'].index(label), 'sources': sources,
            'metadata': {'annotation_origin': 'source_native', **metadata}}


def freeze_confirmation(rows, output_root, provenance, exclusions):
    if not rows or len({r['dataset'] for r in rows}) != 1:
        raise ValueError('nonempty single-dataset confirmation cohort required')
    if len({r['task_id'] for r in rows}) != len(rows):
        raise ValueError('duplicate task IDs')
    dataset = rows[0]['dataset']
    rows = json.loads(json.dumps(sorted(rows, key=lambda r: r['task_id'])))
    for r in rows:
        r['split'] = 'final_holdout'
    # Connected components over article identities (not question IDs).
    parent = list(range(len(rows)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    owner = {}
    for i, row in enumerate(rows):
        for source in row['sources']:
            if source in owner:
                parent[find(i)] = find(owner[source])
            owner[source] = i
    components = {}
    for i, row in enumerate(rows):
        components.setdefault(find(i), []).append(row['task_id'])
    for i, row in enumerate(rows):
        row['group_id'] = 'article-component-' + sha256_obj(sorted(components[find(i)]))[:20]
    destination = Path(output_root)
    visible = destination/'data/normalized'/f'{dataset}.jsonl'
    sealed = destination/'data/sealed'/f'{dataset}.final_holdout.jsonl'
    manifest_path = destination/'data/manifests'/f'{dataset}.json'
    if any(p.exists() for p in (visible, sealed, manifest_path)):
        raise FileExistsError('confirmation cohort cannot overwrite an existing freeze')
    ids = [r['task_id'] for r in rows]
    for p in (visible, sealed, manifest_path):
        p.parent.mkdir(parents=True, exist_ok=True)
    visible.write_text('', encoding='utf-8')
    sealed.write_text(''.join(json.dumps(r, ensure_ascii=False, sort_keys=True)+'\n' for r in rows), encoding='utf-8')
    manifest = {'schema': 'independent-binary-confirmation-v1', 'dataset': dataset,
        'n_questions': len(rows), 'source_file': provenance['source_file'],
        'source_sha256': provenance['source_sha256'], 'source': provenance,
        'salt': 'fixed-preselected-candidate-confirmation-20261019',
        'split_algorithm': 'all-independent-confirmation-no-optimization',
        'ratios': {'dev_optimize': 0, 'selection_gate': 0, 'final_holdout': 100},
        'exclusions': exclusions, 'article_components': len(components),
        'class_counts': dict(Counter(r['options'][r['correct_index']] for r in rows)),
        'normalized_sha256': sha256_file(visible), 'sealed_sha256': sha256_file(sealed),
        'splits': {s: {'n': len(ids) if s=='final_holdout' else 0,
                       'ids': ids if s=='final_holdout' else [],
                       'ids_sha256': sha256_obj(ids if s=='final_holdout' else [])}
                   for s in SPLITS},
        'claims': {'source_native_binary_subset_only': True,
                   'model_pretraining_contamination_ruled_out': False,
                   'no_prompt_adaptation_on_this_cohort': True}}
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n', encoding='utf-8')
    return manifest
