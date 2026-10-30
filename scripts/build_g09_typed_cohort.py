"""Build and preflight the frozen G09 typed cohort without model/network calls.

The source pool is an immutable manifest; this entrypoint only constructs task
inputs. Rubric authoring, answer generation and AI annotation remain separate
commands and are deliberately not invoked here.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.eval.open_research_cohort import build_typed_unannotated_cohort
from src.util import sha256_obj
DEFAULT_POOL=ROOT/'.workbuddy/g09-topic-source-pool-20261018-v26-effective/source-pool.json'
DEFAULT_OUT=ROOT/'.workbuddy/g09-topic-source-cohort-20261018-v26-effective'
def load(p): return json.loads(Path(p).read_text(encoding='utf8'))


def validate_pool_manifest(pool: dict) -> None:
 """Reject an edited source-pool manifest before building any cohort.

 ``pool_sha256`` is computed over the complete manifest excluding the
 self-referential hash field. Checking this identity at the CLI boundary
 keeps the frozen source provenance meaningful for both ``preflight`` and
 ``build``.
 """
 if not isinstance(pool, dict):
  raise ValueError('source pool manifest must be an object')
 expected = pool.get('pool_sha256')
 if (not isinstance(expected, str) or len(expected) != 64 or
     any(ch not in '0123456789abcdef' for ch in expected.lower())):
  raise ValueError('source pool manifest is missing a valid pool_sha256')
 body = {key: value for key, value in pool.items() if key != 'pool_sha256'}
 if sha256_obj(body) != expected.lower():
  raise ValueError('source pool manifest hash does not match its contents')
def specs(pool):
 by={}
 for row in pool['records']: by.setdefault(row['topic_id'],[]).append(row)
 out=[]
 for tid in sorted(by):
  rows=sorted(by[tid],key=lambda r:r['doi'])
  if len(rows)!=2: raise ValueError(f'topic {tid} does not have exactly two records')
  out.append({'group_id':'g09-topic-'+tid,'topic':tid,'source_dois':[r['doi'] for r in rows], 'focal_doi':rows[0]['doi'],'insufficient_target':'G09_UNREPORTED_DETAIL_'+tid.replace('-','_')})
 return out
def build(pool):
 result=build_typed_unannotated_cohort(pool['records'],specs(pool),revision='g09-v26-effective-typed-cohort-r1',source_capture_at='2026-10-18T00:00:00+08:00',split_seed='g09-effective-v26-split-20261018',min_selection_groups=12,min_groups_per_split=12,min_tasks_per_type_per_split=1,license_policy='personal_project_provenance_only')
 return result
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('command',choices=('preflight','build')); ap.add_argument('--pool',type=Path,default=DEFAULT_POOL); ap.add_argument('--out',type=Path,default=DEFAULT_OUT); a=ap.parse_args(); pool=load(a.pool); validate_pool_manifest(pool)
 if pool.get('schema')!='g09-effective-source-pool-v26-r2' or pool.get('complete_topic_groups',0)<48: raise SystemExit('source pool manifest is not valid for typed cohort')
 result=build(pool)
 summary={'status':'zero_call_preflight_passed' if a.command=='preflight' else 'typed_cohort_built','network_calls':0,'model_calls':0,'topic_groups':result['report']['topic_groups'],'groups_by_split':result['report']['groups_by_split'],'tasks_by_split':result['report']['tasks_by_split'],'final_holdout_read_during_build':False,'rubrics_authored':False,'answers_generated':False}
 if a.command=='build':
  a.out.mkdir(parents=True,exist_ok=True); (a.out/'dataset.json').write_text(json.dumps(result['dataset'],ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf8'); (a.out/'report.json').write_text(json.dumps(result['report'],ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf8'); summary['dataset_sha256']=result['report']['dataset_sha256']; summary['report_sha256']=result['report']['report_sha256']
 print(json.dumps(summary,ensure_ascii=False,sort_keys=True))
if __name__=='__main__': main()
