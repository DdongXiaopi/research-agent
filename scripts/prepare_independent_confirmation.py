"""Freeze new confirmation data for an already selected, immutable candidate.

Only reads downloaded public data. Does not invoke models or inspect model
answers. Provenance lists the exact historical source pools excluded.
"""
from __future__ import annotations

import json
import re
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.closure_cohorts import bioasq_rows, qasper_rows, freeze_confirmation, identity, source_key
from src.util import sha256_file, sha256_obj


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from strings(v)
    elif isinstance(value, list):
        for v in value:
            yield from strings(v)


def load_source(path):
    if path.suffix == '.jsonl':
        # str.splitlines also splits U+2028/U+2029 inside valid JSON strings.
        with path.open(encoding='utf-8') as handle:
            return [json.loads(line) for line in handle if line.strip()]
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    sources = ROOT/'.workbuddy/closure-20261019/sources'
    old_bio = ROOT/'vendor/BioASQ2025-UNITOR-main/BioASQ2025-UNITOR-main/data/dataset/answer_generation/train_set_answGEN.json'
    old_scholar = ROOT/'vendor/ScholarQABench-main/ScholarQABench-main/data'
    files = ([old_bio] + sorted(old_scholar.rglob('*.json')) + sorted(old_scholar.rglob('*.jsonl'))
             + [p for p in sorted((ROOT/'data/normalized').glob('*.jsonl'))
                if 'independent_confirmation' not in p.name and 'binary_transfer' not in p.name]
             + sorted((ROOT/'data/manifests').glob('*pubmedqa*.json')))
    text_ids, paper_ids, old_qids, old_documents = set(), set(), set(), set()
    for p in files:
        values = load_source(p)
        if p == old_bio:
            old_qids.update(str(r['id']) for r in values)
        for text in strings(values):
            for piece in [text] + text.splitlines():
                cleaned = re.sub(r'\[/?(?:BS|ES)\]', '', piece, flags=re.I)
                text_ids.add(identity(cleaned))
            paper_ids.update(re.findall(r'(?:arxiv[:/]|arxiv\.org/(?:abs|pdf)/)(\d{4}\.\d{4,5})', text))
            old_documents.update('pmid:'+pid for pid in re.findall(
                r'(?:pubmed/|pubmed\.ncbi\.nlm\.nih\.gov/)(\d+)', text))
            if 'pubmedqa' in p.name:
                match = re.fullmatch(r'(?:pubmedqa_threeway_probe-)?(\d{6,9})', text)
                if match:
                    old_documents.add('pmid:'+match[1])
    exposure = {'policy': 'exclude entire historical source pools, not just sampled task IDs',
                'files': {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in files},
                'normalized_text_id_count': len(text_ids), 'paper_id_count': len(paper_ids),
                'pubmed_identity_count': len(old_documents)}
    exposure['sha256'] = sha256_obj(exposure)
    raw_bio = json.loads((sources/'bioasq14-yesno.json').read_text(encoding='utf-8'))
    bio, bexc = bioasq_rows(raw_bio, old_qids, text_ids, old_documents)
    with tarfile.open(sources/'qasper-test-v0.3.tgz') as archive:
        member = archive.getmember('qasper-test-v0.3.json')
        if member.size > 32_000_000:
            raise ValueError('QASPER JSON exceeds declared read bound')
        raw_qasper = json.load(archive.extractfile(member))
    qasper, qexc = qasper_rows(raw_qasper, text_ids, paper_ids, max_chars=40000)
    if len(bio) < 50 or len(qasper) < 50:
        raise ValueError('insufficient untouched source capacity; do not relax exposure gates')
    download = json.loads((sources/'provenance.json').read_text(encoding='utf-8'))
    report = {'historical_exposure': exposure, 'cohorts': []}
    for rows, exclusions, name in [(bio, bexc, 'bioasq14-yesno.json'),
                                  (qasper, qexc, 'qasper-test-v0.3.tgz')]:
        provenance = {'source_file': (sources/name).relative_to(ROOT).as_posix(),
                      'source_sha256': sha256_file(sources/name),
                      'download_provenance': download,
                      'historical_exposure_sha256': exposure['sha256'],
                      'reference_origin': 'source_native',
                      'candidate_preselected_before_data_acquisition': 'rel-cand-a8511b1481f4',
                      'candidate_parent': 'task-input-baseline-v2'}
        report['cohorts'].append(freeze_confirmation(rows, ROOT, provenance, exclusions))
    destination = sources.parent/'cohort-audit.json'
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'audit': str(destination), 'datasets': [
        {k: m[k] for k in ['dataset','n_questions','article_components','class_counts','exclusions']}
        for m in report['cohorts']]}, indent=2))


if __name__ == '__main__':
    main()
