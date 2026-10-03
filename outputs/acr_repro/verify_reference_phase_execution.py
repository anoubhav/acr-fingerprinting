"""Independently derive all phase counts and validate physical-grid/cohort scope."""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-dir',type=Path,required=True)
    p.add_argument('--results',type=Path,required=True);p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--baseline-phase0',type=Path,required=True)
    p.add_argument('--dense-baseline',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();sys.path.insert(0,str(a.source_dir.resolve()))
    import numpy as np
    from acr_fp import Fingerprinter
    from run_acr_protocol import query_crop
    from run_ablations import ref_path
    from summarize_results import select_threshold,crossed_interval
    plan_path=a.results/'plan.json';plan=json.loads(plan_path.read_text());plan_hash=sha(plan_path)
    if plan['phases']!=list(range(8)) or plan['phase_selection'] is not None or not plan['all_phases_mandatory']:raise ValueError('Incomplete or outcome-selected phase plan')
    for path,key in [(a.protocol,'protocol_sha256'),(a.model,'pca_model_sha256'),(a.baseline_phase0,'phase0_canonical_result_sha256')]:
        if sha(path)!=plan[key]:raise ValueError(f'Frozen {key} changed')
    protocol=json.loads(a.protocol.read_text());model=Fingerprinter.load(a.model)
    source={q['query_id']:q for q in [query_crop(q,5) for q in protocol['queries'] if q['evaluate']]}
    gallery=[r for r in protocol['references'] if r['role'] in ['calibration_known','test_known']]
    gallery_ids=[r['reference_id'] for r in gallery]
    if gallery_ids!=plan['gallery_reference_ids'] or set(source)!=set(plan['query_ids']):raise ValueError('Cohort or gallery changed')
    # Cache digests are the original execution inputs. Independently count residues from physical times.
    reference_counts=np.zeros(8,dtype=np.int64);duration_total=0.;residue_rows=[]
    ref_cache_paths=[path for path in plan['cache_sha256'] if not Path(path).name.startswith('query_')]
    if len(ref_cache_paths)!=len(gallery):raise ValueError('Reference cache ledger scope differs')
    mapping=dict(zip(ref_cache_paths,gallery))
    for path,digest in plan['cache_sha256'].items():
        if sha(path)!=digest:raise ValueError('Frozen feature cache changed')
        if Path(path).name.startswith('query_'):continue
        reference=mapping[path]
        if ref_path(Path(path).parent,reference,model.config).resolve()!=Path(path).resolve():raise ValueError('Cache path does not bind declared source')
        with np.load(path,allow_pickle=False) as z:
            times=z['times'];duration=float(z['duration_s']);duration_total+=duration
        if not np.isfinite(times).all() or (len(times)>1 and np.any(np.diff(times)<=0)):raise ValueError('Cached physical times are not finite and strictly monotonic')
        frames=np.rint((times-model.config.first_center_seconds)/(model.config.hop_length/model.config.sample_rate)).astype(np.int64)
        rebuilt=model.config.first_center_seconds+frames*model.config.hop_length/model.config.sample_rate
        residual=float(np.max(np.abs(times-rebuilt))) if len(times) else 0.
        if residual>1e-8 or np.any(frames<0) or len(np.unique(frames))!=len(frames) or (len(frames)>1 and np.any(np.diff(frames)<=0)):raise ValueError('Invalid physical original-grid reconstruction, nonnegative/unique/monotonic frame indices')
        counts=np.bincount(frames%8,minlength=8);reference_counts+=counts
        residue_rows.append({'reference_id':reference['reference_id'],'role':reference['role'],'feature_cache_sha256':digest,'audio_duration_s':duration,'surviving_frames':len(frames),'original_residue_counts':counts.tolist(),'maximum_grid_reconstruction_residual_s':residual,'original_frames_nonnegative_unique_strictly_monotonic':True})
    phase0=json.loads(a.baseline_phase0.read_text());phase0_lookup={r['query_id']:r for r in phase0['predictions']}
    dense=json.loads(a.dense_baseline.read_text());dense_lookup={r['query_id']:r for r in dense['predictions']}
    if dense['protocol_sha256']!=plan['protocol_sha256'] or dense['factor']!=1 or dense['config']!=asdict(model.config) or set(dense_lookup)!=set(source) or len(dense['predictions'])!=835:raise ValueError('Dense paired baseline changed scope/configuration')
    for field,value in [('exact',True),('top_k',5),('tolerance_sec',.12)]:
        if dense['matcher'][field]!=value:raise ValueError('Dense paired matcher configuration changed')
    for identifier,row in dense_lookup.items():
        q=source[identifier]
        if row['source_id']!=q['source_id'] or row['role']!=q['role'] or row['target_reference_id']!=q['reference_id'] or row['duration_s']!=q['duration_s'] or row['expected_reference_start_s']!=q['expected_reference_start_s'] or row['correct_reference']!=(row['reference_id']==q['reference_id']):raise ValueError('Dense pairing metadata/correctness changed')
    dense_threshold=select_threshold([r for r in dense['predictions'] if r['role']=='calibration_unknown'],.01)
    report={'status':'PASS','plan_sha256':plan_hash,'protocol_sha256':sha(a.protocol),'phase_selected':False,'all8_complete':True,
        'no_primary_results_modified':True,'queries_per_phase':835,'gallery_sources':659,'dense_baseline_sha256':sha(a.dense_baseline),'rows':[]}
    for phase in range(8):
        path=a.results/f'phase_{phase}_5s.json';result=json.loads(path.read_text())
        if result['plan_sha256']!=plan_hash or result['reference_phase']!=phase or result['phase_selected'] is not False:raise ValueError('Wrong phase/result provenance')
        ids=[r['query_id'] for r in result['predictions']]
        if ids!=plan['query_ids'] or len(set(ids))!=835:raise ValueError('Missing/duplicate/extra/out-of-order phase predictions')
        if result['stats']['fingerprints']!=int(reference_counts[phase]):raise ValueError('Original-grid residue count differs')
        if abs(result['stats']['reference_audio_duration_s']-duration_total)>1e-6:raise ValueError('Reference duration changed')
        for r in result['predictions']:
            q=source[r['query_id']]
            for field in ['source_id','role','duration_s','expected_reference_start_s']:
                if r[field]!=q[field]:raise ValueError(f'Query metadata changed: {field}')
            if r['target_reference_id']!=q['reference_id'] or (r['reference_id'] is not None and r['reference_id'] not in gallery_ids):raise ValueError('Target/gallery identity changed')
            correct=r['reference_id']==q['reference_id']
            if correct!=r['correct_reference']:raise ValueError('Incorrect stored content label')
            localized=bool(correct and abs(r['offset_s']-q['expected_reference_start_s'])<=2.)
            if localized!=r['localized_correct']:raise ValueError('Incorrect stored localization label')
            if r['reference_id'] is not None and r['score']!=r['vote_fraction']/(1+r['mean_squared_l2']):raise ValueError('Score evidence inconsistent')
            if r['query_fingerprints']!=phase0_lookup[r['query_id']]['query_fingerprints']:raise ValueError('Query thinning changed across phases')
            if phase==0:
                for field in ['reference_id','score','offset_s','correct_reference','localized_correct','query_fingerprints','vote_fraction','mean_squared_l2']:
                    if r[field]!=phase0_lookup[r['query_id']][field]:raise ValueError('Phase0 failed exact canonical conformance')
        calibration=[r for r in result['predictions'] if r['role']=='calibration_unknown'];known=[r for r in result['predictions'] if r['role']=='test_known'];unknown=[r for r in result['predictions'] if r['role']=='test_unknown']
        if (len(calibration),len(known),len(unknown))!=(74,543,71):raise ValueError('Role denominator differs')
        gate=select_threshold(calibration,.01)
        accepted=lambda r:r['reference_id'] is not None and r['score']>=gate
        item={'phase':phase,'fingerprints':int(reference_counts[phase]),'payload_MB_per_hour':reference_counts[phase]*128/duration_total*3600/1e6,
              'threshold':gate,'calibration_false_accepts':sum(accepted(r) for r in calibration),
              'candidate_correct':sum(r['correct_reference'] for r in known),'candidate_localized':sum(r['localized_correct'] for r in known),
              'accepted_correct':sum(accepted(r) and r['correct_reference'] for r in known),
              'accepted_localized':sum(accepted(r) and r['localized_correct'] for r in known),
              'accepted_target_mismatch':sum(accepted(r) and not r['correct_reference'] for r in known),
              'unknown_accepts':sum(accepted(r) for r in unknown),
              'known_candidate_id_changes_vs_phase0':sum(r['reference_id']!=phase0_lookup[r['query_id']]['reference_id'] for r in known),
              'all_candidate_id_changes_vs_phase0':sum(r['reference_id']!=phase0_lookup[r['query_id']]['reference_id'] for r in result['predictions']),
              'raw_paired_difference_vs_dense':crossed_interval([{'source_id':r['source_id'],'source_query_id':r['source_query_id'],
                  'difference':int(r['correct_reference'])-int(dense_lookup[r['query_id']]['correct_reference'])} for r in known],'difference'),
              'accepted_paired_difference_vs_dense':crossed_interval([{'source_id':r['source_id'],'source_query_id':r['source_query_id'],
                  'difference':int(accepted(r) and r['correct_reference'])-int(dense_lookup[r['query_id']]['reference_id'] is not None and dense_lookup[r['query_id']]['score']>=dense_threshold and dense_lookup[r['query_id']]['correct_reference'])} for r in known],'difference'),
              'result_sha256':sha(path)}
        if item['calibration_false_accepts']!=0:raise ValueError('Original74-negative1% calibration policy violated')
        report['rows'].append(item)
    fields=['candidate_correct','candidate_localized','accepted_correct','accepted_localized','accepted_target_mismatch','unknown_accepts','fingerprints','payload_MB_per_hour']
    report['phase_ranges']={field:[min(r[field] for r in report['rows']),max(r[field] for r in report['rows'])] for field in fields}
    report['phase0_exact_conformance_all835']=True
    report['interpretation']='All-phase sensitivity with unchanged models/crops/query density/matcher. Per-phase independent calibration; all phases reported, none selected. Paired intervals condition on fixed PCA/representation/settings.'
    ledger_path=a.output.with_name('reference_residue_counts.json')
    ledger_path.write_text(json.dumps({'scope':'Derived from original physical timestamps of all659 hashed reference caches; gates do not collapse original frame indices','plan_sha256':plan_hash,'rows':residue_rows,'phase_totals':reference_counts.tolist(),'reference_audio_duration_s':duration_total},indent=2)+'\n')
    report['reference_residue_ledger_sha256']=sha(ledger_path)
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'ranges':report['phase_ranges'],'phase0_exact_conformance_all835':True},indent=2))

if __name__=='__main__':main()
