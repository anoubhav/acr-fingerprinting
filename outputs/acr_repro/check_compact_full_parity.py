"""Full frozen-cohort decision conformance; not model/threshold selection."""
from pathlib import Path
import argparse
import json
import hashlib
import sys
import numpy as np
import faiss
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'outputs/acr_repro'))
from acr_fp import Fingerprinter
from run_acr_protocol import query_crop
from run_ablations import query_path
from study_query_ivf import thin_query
from compact_streaming_index import CompactTemporalIndex

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--experiment',type=Path,default=ROOT/'work/benchmarks/compact_store')
p.add_argument('--protocol',type=Path,default=ROOT/'work/benchmarks/hard_medium_protocol.json')
p.add_argument('--historic-result',type=Path,default=ROOT/'work/benchmarks/acr_ivf_efficiency_study/acr_ivf_selected_5s.json')
a=p.parse_args()
OUTPUT=a.experiment
plan=json.loads((OUTPUT/'plan.json').read_text())
model=Fingerprinter.load(plan['model_path'])
faiss.omp_set_num_threads(4)

with np.load(OUTPUT/'compact_659_metadata.npz',allow_pickle=False) as z:
    ivf=faiss.read_index(str(OUTPUT/'compact_659_r0.index'));ivf.nprobe=16
    compact=CompactTemporalIndex(ivf,z['frame_indices'],z['track_ends'],z['content_ids'],model.config)
# Use independently saved original per-row codes and float64 clocks for baseline.
with np.load(OUTPUT/'baseline_659_metadata.npz',allow_pickle=False) as z:
    ivf=faiss.read_index(str(OUTPUT/'baseline_659_r0.index'));ivf.nprobe=16
    baseline=CompactTemporalIndex(ivf,compact.frame_indices,compact.track_ends,z['content_ids'],model.config)
    baseline.content_codes=z['content_codes'].copy();baseline.times=z['times'].copy()
protocol=json.loads(a.protocol.read_text())
published_path=a.historic_result
published=json.loads(published_path.read_text());previous={q['query_id']:q for q in published['predictions']}
rows=[]
for original in protocol['queries']:
    if not original['evaluate']:continue
    query=query_crop(original,5)
    with np.load(query_path(plan['cache'],query,model.config),allow_pickle=False) as z:
        x,t=model.transform(z['features']),z['times'].copy()
    x,t=thin_query(x,t,4,model.config)
    lhs=baseline.match(x,t,top_k=5,tolerance_sec=.12,limit=10)
    rhs=compact.match(x,t,top_k=5,tolerance_sec=.12,limit=10)
    left,right=(lhs[0] if lhs else {}),(rhs[0] if rhs else {})
    prior=previous[query['query_id']]
    fields=(('content_id','reference_id'),('score','score'),('offset_sec','offset_s'))
    historic={now:left.get(now)==prior.get(old) for now,old in fields}
    rows.append({'query_id':query['query_id'],'role':query['role'],'all_top10_candidates_identical':lhs==rhs,
        'baseline_hit':left,'compact_hit':right,'matches_frozen_prediction':historic,
        'old_threshold_decision_identical':(left.get('score',0)>=published['frozen_threshold'])==prior['accepted']})
    if len(rows)%100==0:print(f'Conformance {len(rows)}/835',flush=True)
report={'scope':'All original835 queries, unchanged factors8/querystride4/IVF512,nprobe16/top5/vote tolerance0.12; conformance only.',
    'protocol_sha256':hashlib.sha256(a.protocol.read_bytes()).hexdigest(),
    'historic_prediction_sha256':hashlib.sha256(published_path.read_bytes()).hexdigest(),
    'all_top10_candidates_exactly_identical':all(r['all_top10_candidates_identical'] for r in rows),
    'all_frozen_top1_ids_scores_offsets_exactly_identical':all(all(r['matches_frozen_prediction'].values()) for r in rows),
    'all_original_threshold_decisions_identical':all(r['old_threshold_decision_identical'] for r in rows),
    'query_count':len(rows),'rows':rows}
(OUTPUT/'full835_parity.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)
if not all(report[k] for k in ('all_top10_candidates_exactly_identical','all_frozen_top1_ids_scores_offsets_exactly_identical','all_original_threshold_decisions_identical')):
    raise AssertionError('Compact store changed frozen decisions')
