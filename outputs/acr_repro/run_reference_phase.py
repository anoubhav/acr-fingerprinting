"""Full, outcome-independent exact reference-grid phase audit.

Retain physical original frame residues before silence filtering, using existing
source/clock-preserving raw caches. Run every phase; no phase is selected. This
is a recognition sensitivity, not a timing benchmark.
"""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import platform
import sys
import time


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save(path, value):
    temp=path.with_suffix(path.suffix+'.partial')
    temp.write_text(json.dumps(value, indent=2)+'\n')
    temp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir',type=Path,required=True)
    p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--baseline-phase0',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--threads',type=int,default=3)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if not 1<=a.threads<=4:raise ValueError('Use one to four CPU threads')
    sys.path.insert(0,str(a.source_dir.resolve()))
    import numpy as np
    import faiss
    from threadpoolctl import threadpool_limits
    from acr_fp import Fingerprinter,ExactIndex,audio_decoder_info
    from run_acr_protocol import query_crop
    from run_ablations import ref_path,query_path
    from summarize_results import summarize,crossed_interval
    threadpool_limits(1,user_api='blas');faiss.omp_set_num_threads(a.threads)
    protocol=json.loads(a.protocol.read_text());model=Fingerprinter.load(a.model)
    references=[r for r in protocol['references'] if r['role'] in ('calibration_known','test_known')]
    queries=[query_crop(q,5) for q in protocol['queries'] if q['evaluate']]
    if len(references)!=659 or len(queries)!=835:raise ValueError('Original full cohort changed')
    model.assert_disjoint([r['reference_id'] for r in references],[q['reference_id'] for q in queries])
    reference_files=[ref_path(a.cache,r,model.config) for r in references]
    query_files=[query_path(a.cache,q,model.config) for q in queries]
    if any(not f.exists() for f in reference_files+query_files):raise ValueError('Existing cache missing; do not silently re-extract')
    print('Hashing frozen reference/query feature cache inputs',flush=True)
    ledger={str(f.resolve()):sha(f) for f in reference_files+query_files}
    source_hashes={n:sha(a.source_dir/n) for n in ['acr_fp.py','run_acr_protocol.py','run_ablations.py','summarize_results.py']}
    plan={'schema_version':1,'experiment':'Full exact factor8 original-frame residue sensitivity',
        'phases':list(range(8)),'factor':8,'duration_s':5.,'queries':835,
        'query_role_counts':dict(Counter(q['role'] for q in queries)),
        'gallery_sources':659,'gallery_reference_ids':[r['reference_id'] for r in references],
        'query_ids':[q['query_id'] for q in queries],
        'query_policy':'Dense original usable frames; no query thinning',
        'selection':'Recover original s from each cache physical timestamp; keep s mod8 == phase; do not thin by surviving-row ordinal. Equivalent to choosing original frames before the unchanged independent silence gates.',
        'pca_refit':False,'phase_selection':None,'all_phases_mandatory':True,
        'matcher':{'exact':True,'top_k':5,'tolerance_sec':.12,'localization_tolerance_s':2.,'threads':a.threads,
          'score':'vote_fraction/(1+mean_squared_L2)'},
        'calibration':'Separate gate for each phase from the same74 calibration-unknown scores; nextafter above order statistic at empirical1% target. No test gate/phase selection.',
        'comparison':'Phase0 full cohort equivalence to canonical primary result before any other phase. Report all phase results/range, IDs, accepted correctness, target mismatches and unknown accepts.',
        'protocol_sha256':sha(a.protocol),'pca_model_sha256':sha(a.model),
        'phase0_canonical_result_sha256':sha(a.baseline_phase0),
        'runner_sha256':sha(__file__),'source_sha256':source_hashes,
        'cache_sha256':ledger,'config':asdict(model.config),
        'timing_claim':False,'environment':{'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'faiss':faiss.__version__,'audio_decoder':audio_decoder_info()}}
    plan_path=a.output/'plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text())!=plan:raise ValueError('Frozen plan/input/source hash changed; use a fresh output directory')
    if not plan_path.exists():save(plan_path,plan)
    plan_hash=sha(plan_path)
    baseline=json.loads(a.baseline_phase0.read_text())
    if baseline['protocol_sha256']!=plan['protocol_sha256']:raise ValueError('Canonical phase0 protocol mismatch')
    baseline_by_id={r['query_id']:r for r in baseline['predictions']}
    if set(baseline_by_id)!={q['query_id'] for q in queries}:raise ValueError('Canonical phase0 cohort differs')
    # Cache only small projected query features; references are read one track at a time.
    features={}
    for q,f in zip(queries,query_files):
        with np.load(f,allow_pickle=False) as z:
            features[q['query_id']]=(model.transform(z['features']),z['times'].copy())
    results=[]
    for phase in range(8):
        result_path=a.output/f'phase_{phase}_5s.json'
        checkpoint_path=a.output/f'phase_{phase}_checkpoint.json'
        if result_path.exists():
            result=json.loads(result_path.read_text())
            if result['plan_sha256']!=plan_hash:raise ValueError('Existing phase output does not match plan')
        else:
            entries=[];duration_total=0.;frame_count=0
            for r,f in zip(references,reference_files):
                with np.load(f,allow_pickle=False) as z:
                    raw=z['features'];times=z['times'].copy();duration_total+=float(z['duration_s'])
                    original=np.rint((times-model.config.first_center_seconds)/(model.config.hop_length/model.config.sample_rate)).astype(np.int64)
                    if len(original) and np.max(np.abs(times-(model.config.first_center_seconds+original*model.config.hop_length/model.config.sample_rate)))>1e-8:raise ValueError('Physical original-grid recovery failed')
                    if len(np.unique(original))!=len(original):raise ValueError('Original frame indices are not unique')
                    keep=original%8==phase
                    # Preserve exact canonical projection arithmetic by projecting the full track before selecting its rows.
                    projected=model.transform(raw)
                    entries.append((r['reference_id'],projected[keep],times[keep]));frame_count+=int(keep.sum())
            index=ExactIndex.from_entries(entries,factor=8,phase=phase,
               grid_step_sec=model.config.hop_length/model.config.sample_rate,grid_origin_sec=model.config.first_center_seconds)
            del entries,raw,projected;gc.collect()
            if len(index.vectors)!=frame_count:raise ValueError('Original-frame selection was changed by index construction')
            stats=index.bytes();stats.update(reference_audio_duration_s=duration_total,reference_count=len(references),
                measured_payload_bytes_per_hour=stats['payload_bytes']/duration_total*3600)
            rows=[]
            if checkpoint_path.exists():
                checkpoint=json.loads(checkpoint_path.read_text())
                if checkpoint['plan_sha256']!=plan_hash:raise ValueError('Checkpoint plan mismatch')
                rows=checkpoint['predictions']
                if [r['query_id'] for r in rows]!=[q['query_id'] for q in queries[:len(rows)]]:raise ValueError('Checkpoint changed cohort order')
            for q in queries[len(rows):]:
                x,t=features[q['query_id']];started=time.perf_counter();hits=index.match(x,t,top_k=5,tolerance_sec=.12,limit=1)
                elapsed=time.perf_counter()-started;hit=hits[0] if hits else {};rid=hit.get('content_id');offset=hit.get('offset_sec');correct=rid==q['reference_id']
                row={'query_id':q['query_id'],'base_query_id':q['query_id'].split('@')[0],'source_id':q['source_id'],
                    'source_query_id':q['source_query_id'],'role':q['role'],'duration_s':q['duration_s'],
                    'target_reference_id':q['reference_id'],'reference_id':rid,'score':hit.get('score',0.),'offset_s':offset,
                    'expected_reference_start_s':q['expected_reference_start_s'],'correct_reference':correct,
                    'localized_correct':bool(correct and abs(offset-q['expected_reference_start_s'])<=2.),
                    'query_fingerprints':len(x),'vote_fraction':hit.get('vote_fraction',0.),'mean_squared_l2':hit.get('mean_distance'),
                    'latency_s':elapsed,'timing_note':'Concurrent descriptive execution telemetry only; no timing claim',
                    'annotation':q['annotation'],'overlap':bool(q['overlapping_other_annotations'])}
                if phase==0:
                    old=baseline_by_id[q['query_id']]
                    for field in ['reference_id','score','offset_s','correct_reference','localized_correct','query_fingerprints','vote_fraction','mean_squared_l2']:
                        if row[field]!=old[field]:
                            save(a.output/'phase0_mismatch.json',{'query_id':q['query_id'],'field':field,'new':row[field],'canonical':old[field]})
                            raise ValueError(f'Phase0 does not exactly reproduce canonical result: {q["query_id"]} {field}')
                rows.append(row)
                if len(rows)%50==0:
                    save(checkpoint_path,{'plan_sha256':plan_hash,'phase':phase,'predictions':rows})
                    print(f'Phase {phase}: {len(rows)}/835 queries complete',flush=True)
            result={'method':'ACR-PCA exact reference-grid phase sensitivity','factor':8,'reference_phase':phase,
                'duration_s_requested':5.,'config':asdict(model.config),'matcher':plan['matcher'],'stats':stats,
                'protocol_sha256':plan['protocol_sha256'],'plan_sha256':plan_hash,'predictions':rows,
                'phase0_exact_canonical_equivalence':True if phase==0 else None,'phase_selected':False,'timing_claim':False}
            save(result_path,result)
            if checkpoint_path.exists():checkpoint_path.unlink()
            del index;gc.collect()
        summary=summarize(result,protocol)
        phase0=baseline_by_id
        known=[r for r in result['predictions'] if r['role']=='test_known']
        unknown=[r for r in result['predictions'] if r['role']=='test_unknown']
        summary['reference_phase']=phase
        summary['phase0_candidate_id_changes_all_queries']=sum(r['reference_id']!=phase0[r['query_id']]['reference_id'] for r in result['predictions'])
        summary['phase0_candidate_id_changes_known']=sum(r['reference_id']!=phase0[r['query_id']]['reference_id'] for r in known)
        summary['phase0_candidate_id_changes_unknown']=sum(r['reference_id']!=phase0[r['query_id']]['reference_id'] for r in unknown)
        summary['phase0_paired_raw_difference']=crossed_interval([{'source_id':r['source_id'],'source_query_id':r['source_query_id'],
            'difference':int(r['correct_reference'])-int(phase0[r['query_id']]['correct_reference'])} for r in known],'difference')
        summary['result_sha256']=sha(result_path);results.append(summary)
        save(a.output/'summary_partial.json',{'plan_sha256':plan_hash,'rows':results,'complete_phases':[r['reference_phase'] for r in results],'all8_complete':len(results)==8,'phase_selected':False})
        print(f'Phase {phase} complete: raw {summary["test_known"]["all"]["correct_reference"]["successes"]}/543, accepted {summary["test_known"]["all"]["accepted_correct"]["successes"]}, unknown {summary["unknown_test"]["false_accepts"]}/71; gate {summary["threshold"]:.10g}',flush=True)
    save(a.output/'summary.json',{'plan_sha256':plan_hash,'rows':results,'complete_phases':list(range(8)),'all8_complete':True,'phase_selected':False,
       'phase0_equivalence':'All835 candidate IDs, scores, offsets, correctness, localization, usable query counts, vote fractions and squared-L2 evidence match canonical original exactly.',
       'interpretation':'Post-hoc declared all-phase sensitivity; no selection or replacement of primary phase0; each phase separately calibrated on the same original74 unknowns. All confidence intervals condition on the fixed model and configuration.'})
    print('All eight phases complete; no phase selected',flush=True)

if __name__=='__main__':main()
