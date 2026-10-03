"""Outcome-independent source/index/numeric checks for frozen acoustic transfer.

The ten probe sources are fixed by metadata-ID hashes, never by recognition
outcome. Diagnostics neither fit a model nor alter the frozen evaluation.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
from scipy.signal import correlate, fftconvolve
from acr_fp import Fingerprinter, load_audio, standardize_rows
from run_sdrr_frozen import build_index
from run_ablations import query_path
from run_acr_protocol import query_crop

def oracle(y):
    import librosa
    mel=librosa.feature.melspectrogram(y=np.pad(y,(512,512)),sr=8000,n_fft=1024,
        hop_length=186,n_mels=32,fmin=315,fmax=1960,center=False,power=1).T
    means=np.lib.stride_tricks.sliding_window_view(mel,32,axis=0).mean(axis=2)
    keep=(means.std(axis=1)>1e-6)&(means.mean(axis=1)>1e-3)
    means=means[keep];diff=np.diff(means,axis=1)
    def z(a):
        std=a.std(axis=1,keepdims=True)
        return np.divide(a-a.mean(axis=1,keepdims=True),std,out=np.zeros_like(a),where=std>1e-8)
    return np.hstack([z(means),z(diff)])

def main():
    p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    import faiss
    faiss.omp_set_num_threads(1)
    protocol=json.loads(a.protocol.read_text());model=Fingerprinter.load(a.model)
    refs=protocol['references']
    chosen=sorted(refs,key=lambda r:hashlib.sha256(('sdrr-integrity-v1:'+r['reference_id']).encode()).hexdigest())[:10]
    index,_=build_index(refs,model,a.cache,8,'exact',None)
    rows=[]
    for reference in chosen:
        original=next(q for q in protocol['queries'] if q['reference_id']==reference['reference_id'])
        q=query_crop(original,10)
        ref,sr=load_audio(reference['path'],8000);phone,_=load_audio(q['path'],8000)
        clean=ref[round(q['expected_reference_start_s']*sr):round((q['expected_reference_start_s']+10)*sr)]
        x,t=model.extract(clean,sr);hit=index.match(x,t,top_k=5,tolerance_sec=.12,limit=1)[0]
        with np.load(query_path(a.cache,q,model.config),allow_pickle=False) as z:
            phone_x,phone_t=model.transform(z['features']),z['times']
            cached_raw=z['features'].copy()
        # Ground-truth alignment is diagnostic only; it never enters retrieval.
        target=np.flatnonzero(index.content_codes==index.content_ids.index(reference['reference_id']))
        tt=index.times[target]
        expected=q['expected_reference_start_s']+phone_t
        near=np.searchsorted(tt,expected)
        near=np.clip(near,1,len(tt)-1)
        near-=np.abs(tt[near-1]-expected)<np.abs(tt[near]-expected)
        true_vectors=index.vectors[target[near]]
        true_dist=np.sum((phone_x-true_vectors)**2,axis=1)
        d,ii=index.search(phone_x,1)
        inliers=index.content_codes[ii[:,0]]==index.content_ids.index(reference['reference_id'])
        raw=np.ascontiguousarray(phone,dtype=np.float32)
        expected_features=oracle(raw)
        actual,_=Fingerprinter().raw_features(raw,8000)
        oracle_error=float(np.max(np.abs(actual-expected_features)))
        np.testing.assert_allclose(actual,expected_features,atol=8e-6,rtol=1e-4)
        np.testing.assert_array_equal(cached_raw,actual)
        start=round(q['expected_reference_start_s']*sr);slack=round(.2*sr)
        window=ref[start-slack:start+len(phone)+slack]
        centered=phone-phone.mean()
        c=correlate(window,centered,mode='valid',method='fft')
        energy=fftconvolve(window.astype(np.float64)**2,np.ones(len(phone)),mode='valid')
        correlation=c/np.sqrt(np.maximum(energy*np.sum(centered.astype(np.float64)**2),1e-20))
        best=int(np.argmax(np.abs(correlation)))
        rows.append({'reference_id':reference['reference_id'],'source_id':reference['source_id'],
            'query_id':q['query_id'],'recording_id':q['recording_id'],'qc_expected_rank':q['qc_expected_rank'],
            'clean_self_match_id':hit['content_id'],'clean_self_match_correct':hit['content_id']==reference['reference_id'],
            'clean_self_match_offset_error_s':abs(hit['offset_sec']-q['expected_reference_start_s']),
            'clean_self_match_votes':hit['votes'],'clean_fingerprints':len(x),
            'phone_fingerprints':len(phone_x),'cached_features_exactly_match_fresh':True,
            'independent_librosa_oracle_max_absolute_error':oracle_error,
            'median_groundtruth_aligned_squared_l2':float(np.median(true_dist)),
            'median_nearest_impostor_or_target_squared_l2':float(np.median(d[:,0])),
            'nearest_neighbor_true_source_fraction':float(np.mean(inliers)),
            'best_expected_waveform_correlation_within200ms':float(correlation[best]),
            'expected_waveform_correlation_lag_s':(best-slack)/sr})
        print(rows[-1],flush=True)
    result={'selection':'Ten source IDs by SHA256(sdrr-integrity-v1:+reference_id), first query per source; independent of test outcomes.',
        'model_refit':False,'retrieval_settings_changed':False,'oracle_groundtruth_used_in_retrieval':False,
        'all_clean_self_matches_correct':all(r['clean_self_match_correct'] for r in rows),
        'all_clean_self_matches_localized_0p1s':all(r['clean_self_match_offset_error_s']<=.1 for r in rows),
        'all_fresh_raw_features_match_cache':True,'all_independent_librosa_numeric_checks_pass':True,'rows':rows,
        'interpretation_scope':'Clean controls test index identity/clocks. Waveform correlation and aligned descriptor distances diagnose paired signal/feature transfer; these are not a new recognition algorithm or parameter search.'}
    a.output.write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
