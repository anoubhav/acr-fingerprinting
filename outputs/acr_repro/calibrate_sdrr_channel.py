"""Post-hoc calibration-only channel-response diagnostic for SD-RR.

Gain fitting uses only known calibration phone/reference pairs. The fitted gain
vector and calibration-unknown gate are frozen before any corrected held-out
query is scored. This leaves the primary frozen representation/results intact.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import numpy as np
from acr_fp import Fingerprinter, load_audio
from run_acr_protocol import query_crop
from run_ablations import query_path
from run_sdrr_frozen import build_index, evaluate_queries, enrich_result
from study_query_ivf import threshold_from_unknown

ROOT=Path(__file__).resolve().parents[2]
FLOOR=1e-6
ALGORITHM='Per-pair centered log(max(reference_whole_crop_mean_mel,1e-6)/max(phone_whole_crop_mean_mel,1e-6)); coordinatewise median across552 known calibration queries; recenter32 log gains to mean0; exponentiate; no clipping/smoothing/weighting.'

def fit_log_gains(ratios):
    values=np.asarray(ratios,dtype=np.float64)
    if values.ndim!=2 or values.shape[1]!=32 or not np.isfinite(values).all():raise ValueError('Invalid calibration ratios')
    centered=values-values.mean(axis=1,keepdims=True)
    gain=np.median(centered,axis=0)
    return gain-gain.mean()

def mel_crop(model,query,path,start):
    y,sr=load_audio(path,model.config.sample_rate)
    begin=round(start*sr);length=round(query['duration_s']*sr)
    audio=y[begin:begin+length]
    if len(audio)!=length:raise ValueError('Calibration/diagnostic crop exceeds decoded physical bounds')
    return model.mel_spectrogram(audio,sr)

def corrected_cache(queries,model,gains,cache,gain_hash):
    cache.mkdir(parents=True,exist_ok=True)
    for i,q in enumerate(queries):
        path=query_path(cache,q,model.config)
        if path.exists():
            with np.load(path,allow_pickle=False) as z:
                if str(z['gain_sha256'])!=gain_hash:raise ValueError('Corrected cache uses another gain vector')
            continue
        mel=mel_crop(model,q,q['path'],q['start_s'])
        corrected=np.asarray(mel*gains[None,:],dtype=np.float32)
        features,times=model.raw_features_from_mels(corrected)
        np.savez_compressed(path,features=features,times=times,gain_sha256=np.array(gain_hash))
        if (i+1)%100==0:print(f'Corrected calibration/declared query features {i+1}/{len(queries)}',flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,default=ROOT/'work/benchmarks/acoustic_rr/sdrr_session_open_protocol.json')
    p.add_argument('--model',type=Path,default=ROOT/'work/benchmarks/acr_medium/pca_model.npz')
    p.add_argument('--baseline',type=Path,default=ROOT/'work/benchmarks/acoustic_rr/acr_session_open/acr_exact_d8_qs1_5s.json')
    p.add_argument('--cache',type=Path,default=ROOT/'work/benchmarks/acoustic_rr/acr_cache_ffmpeg')
    p.add_argument('--output',type=Path,default=ROOT/'work/benchmarks/acoustic_rr/acr_channel_calibration')
    p.add_argument('--prepare-only',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    protocol=json.loads(a.protocol.read_text());model=Fingerprinter.load(a.model)
    known=[query_crop(q,5) for q in protocol['queries'] if q['role']=='calibration_known']
    unknown=[query_crop(q,5) for q in protocol['queries'] if q['role']=='calibration_unknown']
    test=[query_crop(q,5) for q in protocol['queries'] if q['role'].startswith('test')]
    if (len(known),len(unknown),len(test))!=(552,627,309):raise ValueError('Declared calibration cohort changed')
    if hashlib.sha256(a.model.read_bytes()).hexdigest()!=protocol['frozen_model']['sha256']:raise ValueError('PCA hash changed')
    plan={'hypothesis_status':'Post-hoc diagnostic prompted by severe frozen real-acoustic failure; not confirmatory SOTA.',
        'algorithm':ALGORITHM,'floor':FLOOR,'fit_query_ids':[q['query_id'] for q in known],
        'fit_sources':sorted({q['source_id'] for q in known}),'unknown_or_test_audio_used_in_gain_fit':False,
        'gain_fit_source_count':184,'gain_fit_query_count':552,'reference_factor':8,'query_stride':1,'index':'exact',
        'top_k':5,'time_consensus_tolerance_s':.12,'threshold_calibration_unknown':627,'threshold_target':.01,
        'helpfulness_criterion':'Both rawcorrect and acceptedcorrect calibration-known counts strictly improve vs frozen exactD8, each at its independent627-negative1% gate.',
        'heldout_scoring_rule':'Once only after gain+gate freeze and helpfulness criterion passes; otherwise retain calibration-only null.',
        'heldout_known':141,'heldout_unknown':168,'primary_frozen_results_modified':False,'representation_refit':False,
        'pca_refit':False,'query_frontend_gain_calibration':True,
        'protocol_sha256':hashlib.sha256(a.protocol.read_bytes()).hexdigest(),'model_sha256':hashlib.sha256(a.model.read_bytes()).hexdigest()}
    plan_path=a.output/'plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text())!=plan:raise ValueError('Predeclared plan changed')
    plan_path.write_text(json.dumps(plan,indent=2)+'\n')
    if a.prepare_only:return
    import faiss
    faiss.omp_set_num_threads(1)
    baseline=json.loads(a.baseline.read_text())
    if baseline['protocol_sha256']!=plan['protocol_sha256']:raise ValueError('Baseline uses another frozen source split')
    references={r['reference_id']:r for r in protocol['references']}
    ratios=[]
    # Calibration input order is fixed by the frozen protocol, not by scores.
    for i,q in enumerate(known):
        ref=references[q['reference_id']]
        reference=mel_crop(model,q,ref['path'],q['expected_reference_start_s'])
        phone=mel_crop(model,q,q['path'],q['start_s'])
        rmean=reference.mean(axis=0,dtype=np.float64);qmean=phone.mean(axis=0,dtype=np.float64)
        ratios.append(np.log(np.maximum(rmean,FLOOR)/np.maximum(qmean,FLOOR)))
        if (i+1)%50==0:print(f'Fitted paired calibration ratios {i+1}/{len(known)}',flush=True)
    log_gains=fit_log_gains(ratios);gains=np.exp(log_gains).astype(np.float32)
    gain_hash=hashlib.sha256(gains.tobytes()).hexdigest()
    fit={'plan':plan,'log_gains_float64':log_gains.tolist(),'gains_float32':gains.tolist(),'gain_sha256':gain_hash,
        'geometric_mean_float32':float(np.exp(np.log(gains.astype(np.float64)).mean())),
        'calibration_pair_log_ratio_median':np.median(np.asarray(ratios),axis=0).tolist()}
    (a.output/'frozen_gains.json').write_text(json.dumps(fit,indent=2)+'\n')
    corrected=a.output/f'corrected_cache_{gain_hash[:16]}'
    corrected_cache(known+unknown,model,gains,corrected,gain_hash)
    gallery=[r for r in protocol['references'] if r['role'] in ['calibration_known','test_known']]
    index,stats=build_index(gallery,model,a.cache,8,'exact',None)
    cal_rows=evaluate_queries(known+unknown,index,model,corrected,1)
    threshold=threshold_from_unknown([r['score'] for r in cal_rows if r['role']=='calibration_unknown'],target=.01)
    for row in cal_rows:row['accepted']=bool(row['reference_id'] is not None and row['score']>=threshold)
    old=[r for r in baseline['predictions'] if r['role']=='calibration_known']
    now=[r for r in cal_rows if r['role']=='calibration_known']
    before={'raw_correct':sum(r['correct_reference'] for r in old),'accepted_correct':sum(r['correct_reference'] and r['accepted'] for r in old)}
    after={'raw_correct':sum(r['correct_reference'] for r in now),'accepted_correct':sum(r['correct_reference'] and r['accepted'] for r in now)}
    helpful=all(after[key]>before[key] for key in before)
    gate={'gain_sha256':gain_hash,'threshold':threshold,'calibration_unknown':627,'target':.01,
        'calibration_false_accepts':sum(r['accepted'] for r in cal_rows if r['role']=='calibration_unknown'),
        'baseline_calibration_known':before,'corrected_calibration_known':after,'helpfulness_criterion_passed':helpful,
        'heldout_queries_scored_before_freeze':0,'protocol_sha256':plan['protocol_sha256']}
    (a.output/'frozen_gate_and_calibration.json').write_text(json.dumps(gate,indent=2)+'\n')
    result={'method':'ACR-PCA fixed32-band gain diagnostic','dataset':protocol['dataset'],'dataset_doi':protocol['dataset_doi'],
        'closed_set':False,'duration_s_requested':5.,'factor':8,'config':asdict(model.config),
        'matcher':{'top_k':5,'tolerance_sec':.12,'exact':True,'query_stride':1,'threads':1},
        'frozen_threshold':threshold,'stats':stats,'protocol_sha256':plan['protocol_sha256'],
        'provenance':protocol['provenance'],'protocol':protocol['protocol'],'frozen_model_sha256':plan['model_sha256'],
        'channel_diagnostic':fit,'calibration_gate':gate,'predictions':cal_rows,
        'settings_selected_from_test':False,'representation_refit':False,'pca_refit':False,'query_frontend_gain_calibration':True,
        'timing_caveat':'Concurrent jobs; no latency claim; calibration hypothesis is post-hoc.'}
    overlap=json.loads((ROOT/'work/benchmarks/acoustic_rr/source_overlap_audit.json').read_text())
    if helpful:
        corrected_cache(test,model,gains,corrected,gain_hash)
        held=evaluate_queries(test,index,model,corrected,1)
        for row in held:row['accepted']=bool(row['reference_id'] is not None and row['score']>=threshold)
        result['predictions']+=held;result['heldout_evaluation_performed']=True
    else:result['heldout_evaluation_performed']=False
    enrich_result(result,overlap)
    (a.output/'acr_gain_diagnostic_5s.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(gate),flush=True)
    print(json.dumps(result['by_role']),flush=True)

if __name__=='__main__':main()
