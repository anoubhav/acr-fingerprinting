"""Audio-free V2 evidence verification from canonical individual predictions.

Portable exports bind both original execution hashes and sanitized file hashes
through EXPORT_PROVENANCE.json. No media, embeddings or model inference is needed.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib,json,math
from pathlib import Path
import numpy as np
from summarize_results import select_threshold

ROOT=Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def original_sha(path, package_root):
    ledger=package_root/'EXPORT_PROVENANCE.json'
    if ledger.exists():
        relative=str(Path(path).resolve().relative_to(package_root.resolve()))
        record=load(ledger)['files'][relative]
        if sha(path)!=record['export_sha256']:raise ValueError(f'Export SHA mismatch: {relative}')
        return record['source_sha256']
    return sha(path)


def verify_streaming(stream):
    publication=load(stream/'publication_summary.json');cohort=None;worker_total=0
    for measurement in publication['measurements']:
        n=measurement['gallery_sources'];pairs=[]
        for repeat in range(3):
            left=load(stream/f'baseline_{n}_r{repeat}.json');right=load(stream/f'compact_{n}_r{repeat}.json')
            for value in (left,right):
                if value['configuration']!={'factor':8,'query_stride':4,'nlist':512,'nprobe':16,'threads':4}:raise ValueError('Streaming settings changed')
                ids=[q['query_id'] for q in value['queries']]
                if len(ids)!=50 or len(set(ids))!=50:raise ValueError('Streaming cohort')
                if cohort is None:cohort=ids
                if ids!=cohort:raise ValueError('Streaming cohort/order differs')
                if any(len(q['wall_ms'])!=5 or len(q['cpu_ms'])!=5 for q in value['queries']):raise ValueError('Streaming repeats')
                if value['descriptor_sha256']!=measurement['reference_descriptor_sha256']:raise ValueError('Descriptor hash differs')
                expected_bytes=sum(value[k] for k in ['serialized_index_bytes','retained_metadata_bytes','retained_python_vector_bytes','retained_norm_cache_bytes'])
                if expected_bytes!=value['explicit_reference_bytes']:raise ValueError('Explicit-state ledger does not sum')
                query_medians=[float(np.median(q['wall_ms'])) for q in value['queries']]
                if not np.isclose(np.median(query_medians),value['median_match_wall_ms'],rtol=1e-12):raise ValueError('Worker median')
                if not np.isclose(np.quantile(query_medians,.95),value['p95_match_wall_ms'],rtol=1e-12):raise ValueError('Worker p95')
                worker_total+=1
            if [q['hit'] for q in left['queries']]!=[q['hit'] for q in right['queries']]:raise ValueError('Layout changed complete hit dictionaries')
            pairs.append((left,right))
        for index,layout in enumerate(['baseline','compact']):
            for metric,recorded in measurement[layout]['measurements'].items():
                values=[pair[index][metric] for pair in pairs]
                if values!=recorded['values'] or not np.isclose(np.median(values),recorded['median'],rtol=1e-12):raise ValueError(f'Publication aggregate {layout}/{metric}')
            for key in ['explicit_reference_bytes','serialized_index_bytes','retained_metadata_bytes','retained_python_vector_bytes','retained_norm_cache_bytes']:
                if any(pair[index][key]!=measurement[layout][key] for pair in pairs):raise ValueError(f'Publication state {key}')
    if worker_total!=24:raise ValueError('Need24 streaming replicas')
    full=load(stream/'full835_parity.json')
    if full['query_count']!=835 or len(full['rows'])!=835:raise ValueError('Full parity cohort')
    if not all(full[k] for k in ['all_top10_candidates_exactly_identical','all_frozen_top1_ids_scores_offsets_exactly_identical','all_original_threshold_decisions_identical']):raise ValueError('Full parity failed')
    for row in full['rows']:
        if row['baseline_hit']!=row['compact_hit'] or not row['all_top10_candidates_identical'] or not all(row['matches_frozen_prediction'].values()) or not row['old_threshold_decision_identical']:raise ValueError('Full parity row differs')
    return publication,full


def verify_gain(acoustic,root):
    directory=acoustic/'acr_channel_calibration';gains=load(directory/'frozen_gains.json');gate=load(directory/'frozen_gate_and_calibration.json')
    plan=load(directory/'plan.json');protocol=load(acoustic/'sdrr_session_open_protocol.json')
    expected={q['query_id']+'@5s':q for q in protocol['queries'] if q['role']=='calibration_known'}
    if set(plan['fit_query_ids'])!=set(expected) or len(plan['fit_query_ids'])!=552:raise ValueError('Gain fit includes wrong queries')
    fit_sources={q['source_id'] for q in expected.values()}
    if len(fit_sources)!=184 or set(plan['fit_sources'])!=fit_sources:raise ValueError('Gain source cohort')
    if plan['unknown_or_test_audio_used_in_gain_fit'] or plan['pca_refit'] or plan['representation_refit']:raise ValueError('Gain fit scope')
    if gains['plan']!=plan:raise ValueError('Gain plan mismatch')
    array=np.asarray(gains['gains_float32'],dtype=np.float32)
    if array.shape!=(32,) or hashlib.sha256(array.tobytes()).hexdigest()!=gains['gain_sha256']:raise ValueError('Actual gain array SHA')
    if gate['gain_sha256']!=gains['gain_sha256'] or not np.isclose(np.exp(np.mean(np.log(array.astype(float)))),1,atol=1e-6):raise ValueError('Gain metadata')
    if gate['heldout_queries_scored_before_freeze']!=0 or not gate['helpfulness_criterion_passed']:raise ValueError('Pre-heldout freeze rule')
    ledger=load(directory/'freeze_chronology_audit.json')
    for filename,key in [('plan.json','plan_source_sha256'),('frozen_gains.json','gain_file_source_sha256'),('frozen_gate_and_calibration.json','gate_file_source_sha256')]:
        if ledger[key]!=original_sha(directory/filename,root):raise ValueError('Freeze source-byte hash mismatch')
    if not ledger['plan_mtime_ns']<ledger['gain_mtime_ns']<ledger['gate_mtime_ns']<ledger['heldout_cache_first_mtime_ns']:raise ValueError('Freeze chronology')
    heldout={q['query_id']+'@5s':q for q in protocol['queries'] if q['role'].startswith('test')}
    if len(ledger['rows'])!=309 or {r['query_id'] for r in ledger['rows']}!=set(heldout):raise ValueError('Chronology heldout cohort')
    for row in ledger['rows']:
        q=heldout[row['query_id']]
        if row['source_id']!=q['source_id'] or row['role']!=q['role'] or row['duration_s']!=5 or row['start_s']!=2.5 or row['cache_created_mtime_ns']<=ledger['gate_mtime_ns']:raise ValueError('Heldout cache/crop freeze')
    result=load(directory/'acr_gain_diagnostic_5s.json')
    if result['calibration_gate']!=gate or result['channel_diagnostic']!=gains:raise ValueError('Result gained different state/gate')
    metadata=load(acoustic/'acr_result_provenance.json')
    for relative,expected_hash in metadata['result_sha256'].items():
        if original_sha(acoustic/relative,root)!=expected_hash:raise ValueError(f'ACR result provenance hash: {relative}')
    for filename,expected_hash in metadata['code_sha256'].items():
        if sha(root/'outputs/acr_repro'/filename)!=expected_hash:raise ValueError(f'ACR code provenance hash: {filename}')
    return gains,gate,ledger


def prediction_counts(path, protocol_path, package_root):
    result=load(path);protocol=load(protocol_path)
    if result['protocol_sha256']!=original_sha(protocol_path,package_root):
        raise ValueError(f'Original protocol hash mismatch: {path}')
    expected={q['query_id']:q for q in protocol['queries'] if q['evaluate']}
    rows=result['predictions'];ids=[r['query_id'].split('@')[0] for r in rows]
    calibration_only=bool(rows) and all(r['role'].startswith('calibration') for r in rows)
    if calibration_only:expected={qid:q for qid,q in expected.items() if q['role'].startswith('calibration')}
    if len(ids)!=len(set(ids)) or set(ids)!=set(expected):raise ValueError(f'Prediction cohort mismatch: {path}')
    gallery={r['reference_id'] for r in protocol['references'] if r['role'] in protocol['protocol']['gallery_roles']}
    copied=[]
    for row in rows:
        q=expected[row['query_id'].split('@')[0]]
        if (row['role'],row['source_id'],row['target_reference_id'])!=(q['role'],q['source_id'],q['reference_id']):
            raise ValueError(f'Role/source/target changed: {path}:{row["query_id"]}')
        if row['reference_id'] is not None and row['reference_id'] not in gallery:raise ValueError('Candidate outside gallery')
        shift=(q['duration_s']-row['duration_s'])/2
        offset=q['expected_reference_start_s']+shift*q['expected_time_scale']
        if not math.isclose(row['expected_reference_start_s'],offset,abs_tol=1e-8):raise ValueError('Source clock mismatch')
        correct=row['reference_id']==q['reference_id']
        if row['correct_reference']!=correct or not math.isfinite(row['score']):raise ValueError('Incorrect stored label/score')
        if result.get('method','').startswith('SoundFingerprinting') and row['score']!=round(row['raw_score'],12):raise ValueError('MinHash precision mismatch')
        x=dict(row);x['correct']=correct
        x['loc01']=correct and abs(row['offset_s']-offset)<=.1
        x['loc2']=correct and abs(row['offset_s']-offset)<=2
        copied.append(x)
    known=[r for r in copied if r['role']=='test_known'];unknown=[r for r in copied if r['role']=='test_unknown']
    calibration=[r for r in copied if r['role']=='calibration_unknown']
    closed=bool(protocol['protocol'].get('closed_set',False))
    threshold=None if closed else select_threshold(calibration,protocol['protocol'].get('calibration_fpr_target',.01))
    accepted=lambda r:r['reference_id'] is not None and threshold is not None and r['score']>=threshold
    if result.get('frozen_threshold') is not None and not math.isclose(result['frozen_threshold'],threshold,abs_tol=1e-12,rel_tol=1e-12):
        raise ValueError(f'Frozen gate differs from calibration-only derivation: {path}')
    out={'path':str(path),'source_result_sha256':original_sha(path,package_root),'method':result.get('method','ACR-PCA calibration grid'),
      'factor':result.get('factor',1),'duration_s_requested':result.get('duration_s_requested',rows[0]['duration_s']),
      'scope':'calibration_only' if calibration_only else 'full_declared_evaluation',
      'predictions':len(rows),'gallery_references':len(gallery),'test_known':len(known),'test_unknown':len(unknown),
      'raw_correct':sum(r['correct'] for r in known),'raw_localized_0_1':sum(r['loc01'] for r in known),
      'raw_localized_2':sum(r['loc2'] for r in known),'threshold':threshold,
      'accepted_correct':None if closed else sum(accepted(r) and r['correct'] for r in known),
      'accepted_localized_0_1':None if closed else sum(accepted(r) and r['loc01'] for r in known),
      'accepted_localized_2':None if closed else sum(accepted(r) and r['loc2'] for r in known),
      'accepted_target_mismatch':None if closed else sum(accepted(r) and not r['correct'] for r in known),
      'calibration_unknown':len(calibration),'calibration_false_accepts':None if closed else sum(accepted(r) for r in calibration),
      'unknown_false_accepts':None if closed else sum(accepted(r) for r in unknown),'stats':result.get('stats')}
    if calibration:
        allowed=math.floor(protocol['protocol'].get('calibration_fpr_target',.01)*len(calibration))
        if out['calibration_false_accepts']>allowed:raise ValueError('Calibration constraint violated')
    return out


def derive(results, root):
    rows=[]
    for folder in ['acr_medium','acr_default_official_from_extended','acr_ivf_efficiency_study','audfprint_medium',
        'audfprint_calibrated_medium','soundfingerprinting_native_medium','nmfp_results','nmfp_ivf_results',
        'peaknet_primary_results','soundfingerprinting_density_factor2','soundfingerprinting_density_factor4',
        'neural_density_controls/nmfp','neural_density_controls/peaknet','acr_ivf_extended','acr_extended',
        'audfprint_extended','audfprint_calibrated_extended','soundfingerprinting_native_extended',
        'soundfingerprinting_density_factor2_extended','soundfingerprinting_density_factor4_extended',
        'nmfp_extended_results','nmfp_ivf_extended_results','peaknet_extended_results','peaknet_ivf_extended_results']:
        directory=results/folder
        if not directory.is_dir():raise FileNotFoundError(directory)
        for path in sorted(directory.glob('*.json')):
            data=load(path)
            if not isinstance(data,dict) or 'predictions' not in data:continue
            protocol=results/('extended_unknown_protocol.json' if len(data['predictions'])==5835 else 'hard_medium_protocol.json')
            rows.append(prediction_counts(path,protocol,root))
    acoustic=results/'acoustic_rr'
    for folder in ['acr_native','acr_session_open','acr_channel_calibration','audfprint_native','audfprint_candidate',
        'audfprint_session_native','audfprint_session_candidate','soundfingerprinting_native',
        'soundfingerprinting_session_native','nmfp_results','peaknet_results']:
        for path in sorted((acoustic/folder).glob('*.json')):
            data=load(path)
            if not isinstance(data,dict) or 'predictions' not in data:continue
            opened=folder.endswith('session_open') or 'session' in folder or folder=='acr_channel_calibration' or 'session_open' in path.name
            protocol=acoustic/('sdrr_session_open_protocol.json' if opened else 'sdrr_native_protocol.json')
            rows.append(prediction_counts(path,protocol,root))
    stream=results/'streaming_index' if (results/'streaming_index').is_dir() else root/'work/revision2026/streaming_index'
    publication,full=verify_streaming(stream)
    gains,gate,chronology=verify_gain(acoustic,root)
    clock_audit=load(results/'neural_density_controls/clock_compatibility_audit.json')
    if clock_audit['historical_outputs_exported_as_primary'] or not clock_audit['audit_only']:raise ValueError('Obsolete Peak clock output marked primary')
    for record in clock_audit['canonical_results']:
        if original_sha(results/record['canonical_result_path'],root)!=record['canonical_result_sha256']:raise ValueError('Canonical Peak clock audit hash')
        if record['changed_query_count']!=len(record['changes']):raise ValueError('Clock delta count')
    return {'version':'V2','predictions_recomputed_from_current_canonical_rows':rows,
        'streaming_workers':24,'full835_conformance':full,'streaming_publication':publication,
        'gain_fit_queries':gains['plan']['gain_fit_query_count'],'gain_fit_sources':gains['plan']['gain_fit_source_count'],
        'gain_freeze_before_heldout':True,'chronology_heldout_cache_records':len(chronology['rows']),
        'clock_compatibility_delta_audit':clock_audit,
        'canonical_peak_clock':'track-local physical grid; obsolete compatibility outputs excluded'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=ROOT/'results' if (ROOT/'results').is_dir() else ROOT/'work/benchmarks')
    parser.add_argument('--output',type=Path,default=ROOT/'work/v2_evidence_recomputed.json')
    parser.add_argument('--check',type=Path,help='Compare recomputed scientific rows against saved V2 evidence summary')
    args=parser.parse_args()
    manifest=ROOT/'SHA256SUMS.json'
    if manifest.exists():
        for name,expected in load(manifest).items():
            if sha(ROOT/name)!=expected:raise ValueError(f'Artifact checksum mismatch: {name}')
    data=derive(args.results,ROOT)
    if args.check:
        old=load(args.check)
        def numbers(rows):
            return [{k:v for k,v in row.items() if k not in ('path','stats')} for row in rows]
        if numbers(old['predictions_recomputed_from_current_canonical_rows'])!=numbers(data['predictions_recomputed_from_current_canonical_rows']):
            raise ValueError('Canonical V2 count/gate/result hashes changed')
    try:
        from derive_v2_profile_summary import derive as derive_timing,check as check_timing
    except ImportError:
        derive_timing=None
    if derive_timing:
        profiles=args.results/'profiling' if (args.results/'profiling').is_dir() else ROOT/'work/profiling'
        data['current_timing']=derive_timing(profiles)
        check_timing(data['current_timing'],load(profiles/'paper_timing_final.json'))
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(data,indent=2)+'\n')
    print(f"Verified {len(data['predictions_recomputed_from_current_canonical_rows'])} current result cohorts,24 streaming replicas and calibrated32-gain state; no audio/inference required.")


if __name__=='__main__':main()
