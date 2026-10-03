"""Independent NumPy-FFT and direct-Pearson checks of the frozen audit peaks."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-dir',type=Path,required=True)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--summary',type=Path,required=True)
    p.add_argument('--protocol',type=Path,required=True);p.add_argument('--results',type=Path,required=True)
    p.add_argument('--gate',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();sys.path.insert(0,str(a.source_dir.resolve()))
    import numpy as np
    from audit_unknown_matches import decoded
    from summarize_results import select_threshold
    plan=json.loads(a.plan.read_text());summary=json.loads(a.summary.read_text())
    for path,key in [(a.protocol,'protocol_sha256'),(a.results,'result_sha256'),(a.gate,'gate_sha256')]:
        if sha(path)!=plan[key]:raise ValueError('Frozen primary input changed')
    if summary['plan_sha256']!=sha(a.plan) or plan['criterion_absolute_full_crop_pearson_at_least']!=.95 or plan['offset_radius_s']!=.25:raise ValueError('Prespecified criterion changed')
    result=json.loads(a.results.read_text());protocol=json.loads(a.protocol.read_text());gate=json.loads(a.gate.read_text())
    queries={q['query_id']:q for q in protocol['queries']};refs={r['reference_id']:r for r in protocol['references']}
    calibration=[r for r in result['predictions'] if r['role']=='calibration_unknown']
    if select_threshold(calibration,.001)!=gate['threshold'] or gate['threshold']!=result['frozen_threshold']:raise ValueError('Frozen calibration gate differs')
    selected=[r for r in result['predictions'] if r['role']=='test_unknown' and r['reference_id'] is not None and r['score']>=gate['threshold'] and queries[r['query_id'].split('@')[0]].get('negative_population')=='additional_clean_FMA']
    expected={r['query_id']:r for r in selected};planned={r['query_id']:r for r in plan['rows']};recorded={r['query_id']:r for r in summary['rows']}
    if len(expected)!=16 or len(planned)!=16 or len(recorded)!=16 or expected.keys()!=planned.keys() or expected.keys()!=recorded.keys():raise ValueError('Not all16 original frozen false accepts were audited')
    errors=[];rows=[]
    for identifier,prediction in expected.items():
        item=recorded[identifier];frozen=planned[identifier];q=queries[identifier.split('@')[0]];ref=refs[prediction['reference_id']]
        for key,value in [('predicted_reference_id',prediction['reference_id']),('frozen_score',prediction['score']),('predicted_reference_offset_s',prediction['offset_s']),('query_crop_start_s',q['start_s']),('query_crop_duration_s',5.)]:
            if frozen[key]!=value or item[key]!=value:raise ValueError('Frozen audit selection changed')
        for path,key in [(q['path'],'unknown_encoded_source_sha256'),(ref['path'],'reference_encoded_source_sha256')]:
            if sha(path)!=frozen[key] or frozen[key]!=item[key]:raise ValueError('Encoded source digest differs')
        source=decoded(q['path']);reference=decoded(ref['path']);n=40000;start=round(q['start_s']*8000)
        query=source[start:start+n];lower=max(0,round((prediction['offset_s']-.25)*8000));upper=min(len(reference),round((prediction['offset_s']+.25)*8000)+n);window=reference[lower:upper]
        if len(query)!=n:raise ValueError('Not a complete five-second query')
        for array,key in [(query,'query_pcm_crop_sha256'),(window,'reference_pcm_search_window_sha256')]:
            if hashlib.sha256(np.asarray(array,dtype='<f4').tobytes()).hexdigest()!=item[key]:raise ValueError('Actual decoded crop/window digest differs')
        if len(window)<n:
            if item['waveform_audit_status']!='insufficient_full_crop_support':raise ValueError('Unsupported full crop misreported')
            rows.append({'query_id':identifier,'status':'insufficient_full_crop_support','independently_checked':True});continue
        x=window.astype(np.float64);y=query.astype(np.float64);y-=y.mean();normq=np.linalg.norm(y)
        if normq<1e-9:
            if item['waveform_audit_status']!='silent_query':raise ValueError('Constant query not disclosed')
            rows.append({'query_id':identifier,'status':'silent_query','independently_checked':True});continue
        if np.std(x)==0:
            if item['waveform_audit_status']!='constant_reference_search_window':raise ValueError('Constant reference not disclosed')
            rows.append({'query_id':identifier,'status':'constant_reference_search_window','independently_checked':True});continue
        # Independent FFT engine and FFT-computed rolling moments, versus the
        # original SciPy correlation plus cumulative-sum denominator.
        size=1<<(len(x)+n-2).bit_length();fx=np.fft.rfft(x,size);fy=np.fft.rfft(y[::-1],size)
        numerator=np.fft.irfft(fx*fy,size)[n-1:len(x)]
        ones=np.fft.rfft(np.ones(n),size)
        sums=np.fft.irfft(fx*ones,size)[n-1:len(x)]
        sums2=np.fft.irfft(np.fft.rfft(x*x,size)*ones,size)[n-1:len(x)]
        normr=np.sqrt(np.maximum(0,sums2-sums*sums/n));values=np.divide(numerator,normq*normr,out=np.zeros_like(numerator),where=normr>1e-9)
        peak=int(np.argmax(np.abs(values)));maximum=float(abs(values[peak]));offset=(lower+peak)/8000
        aligned=x[peak:peak+n];centered=aligned-aligned.mean();direct=float(np.dot(centered,y)/(np.linalg.norm(centered)*normq))
        if abs(maximum-item['absolute_waveform_correlation'])>1e-8 or abs(offset-item['refined_reference_offset_s'])>1/8000/2 or abs(direct-item['signed_waveform_correlation'])>1e-8:raise ValueError('Independent maximum or lag/direct Pearson differs')
        if item['strong_waveform_overlap_0p95']!=(maximum>=.95):raise ValueError('Prespecified flag misreported')
        errors.append(abs(direct-item['signed_waveform_correlation']))
        rows.append({'query_id':identifier,'status':'measured','numpy_fft_maximum_absolute_pearson':maximum,'numpy_fft_refined_offset_s':offset,'direct_peak_signed_pearson':direct,'independently_checked':True})
    statuses={s:sum(r['status']==s for r in rows) for s in {r['status'] for r in rows}}
    if statuses!=summary['status_counts'] or summary['frozen_false_accept_count']!=16 or summary['clean_test_count']!=4000 or summary['count_or_label_changes']!=0:raise ValueError('Reported status/count scope changed')
    out={'status':'PASS','plan_sha256':sha(a.plan),'summary_sha256':sha(a.summary),'verified_original_clean_false_accepts':16,
        'criterion_unchanged':True,'primary_count_and_labels_unchanged':True,'independent_algorithm':'NumPyFFT for full lag maxima and rolling moments; direct centered dot-product Pearson at selected lag',
        'maximum_direct_pearson_discrepancy':max(errors,default=0),'rows':rows}
    a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='rows'},indent=2))

if __name__=='__main__':main()
