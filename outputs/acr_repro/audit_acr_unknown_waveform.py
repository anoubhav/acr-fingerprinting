"""Label-quality audit of every frozen-q4 expanded clean-unknown acceptance.

No prediction, gate, label or false-accept count is changed. The fixed
absolute-Pearson rule flags a shared five-second waveform only; failure to flag
one does not prove distinct compositions or recordings.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def save(path,data):
    temp=path.with_suffix(path.suffix+'.partial');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir',type=Path,required=True);p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--results',type=Path,required=True);p.add_argument('--gate',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--prepare-only',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);sys.path.insert(0,str(a.source_dir.resolve()))
    import numpy as np
    from summarize_results import select_threshold
    from audit_unknown_matches import aligned_correlation,decoded
    from acr_fp import audio_decoder_info
    protocol=json.loads(a.protocol.read_text());result=json.loads(a.results.read_text());gate=json.loads(a.gate.read_text())
    if result['protocol_sha256']!=sha(a.protocol):raise ValueError('Frozen prediction protocol byte hash changed')
    if result['matcher']['query_stride']!=4 or result['matcher']['nprobe']!=16:raise ValueError('Expected frozenq4/probe16 primary configuration')
    if result['frozen_threshold']!=gate['threshold'] or gate['uses_extended_test_outcomes']:raise ValueError('Frozen gate changed')
    calibration=[r for r in result['predictions'] if r['role']=='calibration_unknown']
    if len(calibration)!=1074 or select_threshold(calibration,.001)!=gate['threshold']:raise ValueError('Original expanded calibration policy changed')
    queries={q['query_id']:q for q in protocol['queries']};references={r['reference_id']:r for r in protocol['references']}
    hits=[r for r in result['predictions'] if r['role']=='test_unknown' and r['reference_id'] is not None and r['score']>=gate['threshold'] and queries[r['query_id'].split('@')[0]].get('negative_population')=='additional_clean_FMA']
    hits.sort(key=lambda r:r['query_id'])
    if len(hits)!=16:raise ValueError('All16 frozen clean false accepts are mandatory')
    scope=[]
    for r in hits:
        q=queries[r['query_id'].split('@')[0]];ref=references[r['reference_id']]
        if q['duration_s']!=5. or r['duration_s']!=5.:raise ValueError('Expected original five-second crop')
        qsha=sha(q['path']);rsha=sha(ref['path'])
        if qsha!=q['audio_sha256']:raise ValueError('Queryencoded file does not match acquisition digest')
        scope.append({'query_id':r['query_id'],'unknown_source_id':q['source_id'],'predicted_reference_id':r['reference_id'],
            'frozen_score':r['score'],'predicted_reference_offset_s':r['offset_s'],
            'query_crop_start_s':q['start_s'],'query_crop_duration_s':q['duration_s'],
            'unknown_encoded_source_sha256':qsha,'reference_encoded_source_sha256':rsha,
            'unknown_encoded_path':q['path'],'reference_encoded_path':ref['path']})
    plan={'scope':'All16 frozen expanded-q4 accepted clean source-ID negatives; label-quality diagnostic only',
        'selection':'Existing frozen primary acceptance, not a newly selected gate/threshold; all16 mandatory, query-ID order',
        'sample_rate':8000,'criterion_absolute_full_crop_pearson_at_least':.95,'offset_radius_s':.25,
        'offset_search':'Complete crop at each sample lag within +/-0.25s of frozen predicted offset; no scale/tempo search',
        'constant_query_norm_threshold':1e-9,'minimum_full_crop_samples':40000,
        'primary_predictions_modified':False,'primary_gate_modified':False,'primary_labels_modified':False,
        'frozen_clean_false_accept_count':16,'frozen_clean_test_count':4000,
        'interpretation':'Aflag describes strong shared five-second waveform, not whole-recording identity. Anegative correlation test cannot prove distinct composition/recording.',
        'protocol_sha256':sha(a.protocol),'result_sha256':sha(a.results),'gate_sha256':sha(a.gate),
        'runner_sha256':sha(__file__),'source_sha256':{n:sha(a.source_dir/n) for n in ['audit_unknown_matches.py','acr_fp.py','summarize_results.py']},
        'decoder':audio_decoder_info(),'rows':scope}
    plan_path=a.output/'plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text())!=plan:raise ValueError('Diagnostic plan/criterion/frozen inputs changed; use a fresh outputdirectory')
    if not plan_path.exists():save(plan_path,plan)
    if a.prepare_only:
        print('Frozenall16 IDs, references, offsets, encoded-source digests and criterion beforedecode',flush=True);return
    plan_hash=sha(plan_path);rows=[]
    for selected in scope:
        r=next(r for r in hits if r['query_id']==selected['query_id']);q=dict(queries[r['query_id'].split('@')[0]]);ref=references[r['reference_id']]
        source=decoded(q['path']);reference=decoded(ref['path']);begin=round(q['start_s']*8000);length=round(q['duration_s']*8000)
        query=source[begin:begin+length]
        if len(query)!=length:raise ValueError('Physical query crop is shorter than frozen plan')
        lower=max(0,round((r['offset_s']-.25)*8000));upper=min(len(reference),round((r['offset_s']+.25)*8000)+length)
        window=reference[lower:upper]
        item=dict(selected);item.update(query_pcm_crop_sha256=hashlib.sha256(np.asarray(query,dtype='<f4').tobytes()).hexdigest(),
            reference_pcm_search_window_sha256=hashlib.sha256(np.asarray(window,dtype='<f4').tobytes()).hexdigest(),
            query_crop_samples=len(query),reference_search_window_samples=len(window),
            full_crop_search_lag_count=max(0,len(window)-len(query)+1),
            query_centered_norm=float(np.linalg.norm(query.astype(float)-float(query.mean(dtype=np.float64)))),
            reference_search_window_std=float(np.std(window.astype(float))) if len(window) else None)
        # The actual correlation and refined lag come from the original existing auditimplementation.
        item.update(aligned_correlation(q,ref,r['offset_s'],radius_s=.25))
        if item['waveform_audit_status']=='measured' and item['reference_search_window_std']==0:
            item['waveform_audit_status']='constant_reference_search_window'
            item['strong_waveform_overlap_0p95']=False
        if item.get('refined_reference_offset_s') is not None:
            aligned=round(item['refined_reference_offset_s']*8000)
            item['matching_reference_crop_sha256']=hashlib.sha256(np.asarray(reference[aligned:aligned+length],dtype='<f4').tobytes()).hexdigest()
            item['refined_lag_from_frozen_offset_s']=item['refined_reference_offset_s']-r['offset_s']
        rows.append(item)
        print(f"Audited {len(rows)}/16: {selected['query_id']} status {item['waveform_audit_status']} absolute correlation {item.get('absolute_waveform_correlation')}",flush=True)
    if sha(a.protocol)!=plan['protocol_sha256'] or sha(a.results)!=plan['result_sha256'] or sha(a.gate)!=plan['gate_sha256']:raise ValueError('Primary input changed during audit')
    for row in scope:
        if sha(row['unknown_encoded_path'])!=row['unknown_encoded_source_sha256'] or sha(row['reference_encoded_path'])!=row['reference_encoded_source_sha256']:raise ValueError('Encoded source changed during audit')
    measured=[r for r in rows if r['waveform_audit_status']=='measured']
    out={'status':'PASS','plan_sha256':plan_hash,'query_count':16,'all16_frozen_accepted_unknowns_audited':True,
        'criterion_absolute_full_crop_pearson_at_least':.95,'offset_radius_s':.25,
        'frozen_false_accept_count':16,'clean_test_count':4000,'count_or_label_changes':0,
        'strong_shared_5s_waveform_flags':sum(r.get('strong_waveform_overlap_0p95',False) for r in rows),
        'status_counts':{status:sum(r['waveform_audit_status']==status for r in rows) for status in sorted({r['waveform_audit_status'] for r in rows})},
        'max_absolute_waveform_correlation':max((r['absolute_waveform_correlation'] for r in measured),default=None),
        'median_absolute_waveform_correlation':float(np.median([r['absolute_waveform_correlation'] for r in measured])) if measured else None,
        'interpretation':plan['interpretation'],'input_byte_hashes_rechecked_after_audit':True,'rows':rows}
    save(a.output/'summary.json',out);print(json.dumps({k:v for k,v in out.items() if k!='rows'},indent=2),flush=True)

if __name__=='__main__':main()
