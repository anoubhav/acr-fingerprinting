"""Frozen FMA-PCA ACR transfer to SD-RR real acoustic recordings.

No representation fitting occurs here. Native closed-set and session-separated
open-set adaptation are distinct protocols; only calibration-unknown scores
choose the latter's rejection threshold. Runtime observations are contended.
"""
from __future__ import annotations
import argparse
import concurrent.futures
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
from acr_fp import Fingerprinter, ExactIndex, audio_decoder_info
from run_acr_protocol import extract_ref, extract_query, query_crop
from run_ablations import ref_path, query_path
from study_query_ivf import thin_query, threshold_from_unknown

ROOT=Path(__file__).resolve().parents[2]
_QUERY_MODEL=None

def init_query_model(model_path):
    global _QUERY_MODEL
    _QUERY_MODEL=Fingerprinter.load(model_path)


def cache_query(task):
    query,cache=task
    path=query_path(cache,query,_QUERY_MODEL.config)
    exists=path.exists()
    if not exists:extract_query(query,_QUERY_MODEL,cache)
    return query['query_id'],exists


def source_bootstrap(rows,field,iterations=5000,seed=20261003,cluster_field='source_id'):
    """Conditional source-cluster interval; not a room/device population bound."""
    if not rows:return None
    groups={}
    for row in rows:groups.setdefault(row[cluster_field],[]).append(int(row[field]))
    sums=np.array([sum(v) for v in groups.values()],dtype=np.int64)
    sizes=np.array([len(v) for v in groups.values()],dtype=np.int64)
    rng=np.random.default_rng(seed)
    draw=rng.integers(0,len(sums),size=(iterations,len(sums)))
    values=sums[draw].sum(axis=1)/sizes[draw].sum(axis=1)
    return [float(v) for v in np.quantile(values,[.025,.975])]


def summarize(rows):
    known=[r for r in rows if r['role'].endswith('known') and not r['role'].endswith('unknown')]
    unknown=[r for r in rows if r['role'].endswith('unknown')]
    out={'query_count':len(rows),'source_count':len({r['source_id'] for r in rows}),
        'creator_count':len({r['creator_group'] for r in rows}),
        'known_queries':len(known),'unknown_queries':len(unknown)}
    for field in ['correct_reference','localized_correct_0p1s','localized_correct_2s']:
        out[field+'_count']=sum(r[field] for r in known)
        out[field+'_rate']=sum(r[field] for r in known)/len(known) if known else None
        out[field+'_source_bootstrap_ci95']=source_bootstrap(known,field)
        out[field+'_creator_bootstrap_ci95']=source_bootstrap(known,field,cluster_field='creator_group')
    if rows and 'accepted' in rows[0]:
        for r in known:
            r['accepted_correct']=bool(r['accepted'] and r['correct_reference'])
            r['accepted_localized_0p1s']=bool(r['accepted'] and r['localized_correct_0p1s'])
            r['accepted_localized_2s']=bool(r['accepted'] and r['localized_correct_2s'])
        for field in ['accepted_correct','accepted_localized_0p1s','accepted_localized_2s']:
            out[field+'_count']=sum(r[field] for r in known)
            out[field+'_rate']=sum(r[field] for r in known)/len(known) if known else None
            out[field+'_source_bootstrap_ci95']=source_bootstrap(known,field)
            out[field+'_creator_bootstrap_ci95']=source_bootstrap(known,field,cluster_field='creator_group')
        out['wrong_accepted_known']=sum(r['accepted'] and not r['correct_reference'] for r in known)
        out['false_accepts']=sum(r['accepted'] for r in unknown)
        out['empirical_false_accept_rate']=out['false_accepts']/len(unknown) if unknown else None
        out['false_accept_source_bootstrap_ci95']=source_bootstrap(unknown,'accepted')
        out['false_accept_creator_bootstrap_ci95']=source_bootstrap(unknown,'accepted',cluster_field='creator_group')
        out['false_accept_source_count']=len({r['source_id'] for r in unknown if r['accepted']})
        out['false_accept_creator_count']=len({r['creator_group'] for r in unknown if r['accepted']})
    return out


def build_index(gallery,model,cache,factor,kind,frozen_ivf):
    """Thin physical-grid raw rows before the unchanged row-wise projection."""
    entries=[];duration=0.
    for reference in gallery:
        with np.load(ref_path(cache,reference,model.config),allow_pickle=False) as z:
            raw,times=z['features'],z['times'].copy();duration+=float(z['duration_s'])
        frames=np.rint((times-model.config.first_center_seconds)/(model.config.hop_length/model.config.sample_rate)).astype(np.int64)
        keep=frames%factor==0
        entries.append((reference['reference_id'],model.transform(raw[keep]),times[keep]))
    index=ExactIndex.from_entries(entries,factor=1)
    index.factor=factor
    del entries;gc.collect()
    centroid_hash=None
    if kind=='ivf':
        import faiss
        ivf=faiss.read_index(str(frozen_ivf))
        if ivf.d!=model.config.output_dim or ivf.nlist!=512:raise ValueError('Frozen IVF dimensions differ')
        ivf.reset();ivf.set_direct_map_type(faiss.DirectMap.NoMap);ivf.nprobe=16
        centroid_hash=hashlib.sha256(faiss.serialize_index(ivf).tobytes()).hexdigest()
        ivf.add(index._search_vectors);index._faiss=ivf
    stats=index.bytes();stats.update(reference_count=len(gallery),reference_audio_duration_s=duration,
        vector_descriptor_sha256=hashlib.sha256(memoryview(index.vectors).cast('B')).hexdigest(),
        empty_frozen_centroid_index_sha256=centroid_hash)
    return index,stats


def evaluate_queries(queries,index,model,cache,stride):
    rows=[]
    for i,q in enumerate(queries):
        with np.load(query_path(cache,q,model.config),allow_pickle=False) as z:
            x,t=model.transform(z['features']),z['times'].copy()
        x,t=thin_query(x,t,stride,model.config)
        hits=index.match(x,t,top_k=5,tolerance_sec=.12,limit=1)
        h=hits[0] if hits else {};rid=h.get('content_id');offset=h.get('offset_sec')
        correct=rid==q['reference_id']
        error=abs(offset-q['expected_reference_start_s']) if offset is not None else None
        rows.append({'query_id':q['query_id'],'base_query_id':q['query_id'].rsplit('@',1)[0],
            'source_id':q['source_id'],'source_query_id':q['source_query_id'],'recording_id':q['recording_id'],
            'creator_group':q['creator_group'],'role':q['role'],'duration_s':q['duration_s'],
            'target_reference_id':q['reference_id'],'reference_id':rid,'score':h.get('score',0.),'offset_s':offset,
            'expected_reference_start_s':q['expected_reference_start_s'],'absolute_offset_error_s':error,
            'correct_reference':correct,'localized_correct_0p1s':bool(correct and error<=.1),
            'localized_correct_2s':bool(correct and error<=2.),'localized_correct':bool(correct and error<=2.),
            'query_fingerprints':len(x),'votes':h.get('votes',0),'vote_fraction':h.get('vote_fraction',0.),
            'mean_squared_l2':h.get('mean_distance'),'qc_expected_rank':q['qc_expected_rank'],
            'alignment_warning':q.get('alignment_warning',''),'annotation':q['annotation']})
        if (i+1)%100==0:print(f'Queried {i+1}/{len(queries)}',flush=True)
    return rows


def enrich_result(result,overlap):
    rows=result['predictions']
    result['summary']=summarize(rows)
    result['by_role']={role:summarize([r for r in rows if r['role']==role]) for role in sorted({r['role'] for r in rows})}
    result['by_recording_session']={session:summarize([r for r in rows if r['recording_id']==session]) for session in sorted({r['recording_id'] for r in rows})}
    result['by_qc_rank']={'expected_top1':summarize([r for r in rows if r['qc_expected_rank']==1]),
        'expected_not_top1':summarize([r for r in rows if r['qc_expected_rank']!=1])}
    excluded={'sdrr:'+v for v in overlap['all_fma_title_matched_sdrr_ids']}
    result['fma_overlap_sensitivity']={'excluded_reference_ids':sorted(excluded),'gallery_changed':False,
        'summary':summarize([r for r in rows if r['target_reference_id'] not in excluded]),
        'caution':overlap['caution']}
    result['uncertainty']='5000 deterministic source-cluster bootstrap draws; conditional on recorded sessions. Four sessions and one declared capture-device pair do not support a device/room-population claim.'


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--model',type=Path,default=ROOT/'work/benchmarks/acr_medium/pca_model.npz')
    p.add_argument('--frozen-ivf',type=Path,default=ROOT/'work/benchmarks/acr_ivf_study/ivf_factor8.index')
    p.add_argument('--cache',type=Path,default=ROOT/'work/benchmarks/acoustic_rr/acr_cache_ffmpeg')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--overlap-audit',type=Path,default=ROOT/'work/benchmarks/acoustic_rr/source_overlap_audit.json')
    p.add_argument('--durations',default='10,5')
    p.add_argument('--settings',default='exact:8:1,ivf:8:4,exact:1:1')
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--threads',type=int,default=2)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);a.cache.mkdir(parents=True,exist_ok=True)
    import faiss
    faiss.omp_set_num_threads(a.threads)
    protocol=json.loads(a.protocol.read_text());model=Fingerprinter.load(a.model)
    model_hash=hashlib.sha256(a.model.read_bytes()).hexdigest()
    if model_hash!=protocol['frozen_model']['sha256']:raise ValueError('Frozen PCA hash differs')
    if not protocol['provenance']['physical_bounds_verified']:raise ValueError('Executable audio bounds not verified')
    gallery=[r for r in protocol['references'] if r['role'] in ['calibration_known','test_known']]
    unknown={q['reference_id'] for q in protocol['queries'] if q['role'].endswith('unknown')}
    if unknown & {r['reference_id'] for r in gallery}:raise ValueError('Unknown reference enrolled')
    model.assert_disjoint([r['reference_id'] for r in gallery],[q['reference_id'] for q in protocol['queries']])
    overlap=json.loads(a.overlap_audit.read_text())
    settings=[]
    for value in a.settings.split(','):
        kind,factor,stride=value.split(':');settings.append((kind,int(factor),int(stride)))
    if any(kind not in ['exact','ivf'] or factor<1 or stride<1 for kind,factor,stride in settings):raise ValueError('Invalid setting')
    tasks=[(r,asdict(model.config),str(a.cache)) for r in gallery]
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
        for i,(_,_,_) in enumerate(pool.map(extract_ref,tasks)):
            if (i+1)%50==0:print(f'Cached references {i+1}/{len(gallery)}',flush=True)
    durations=[float(v) for v in a.durations.split(',')]
    # Native10s sparse exact first, before preparing adapted5s caches.
    for duration in durations:
        queries=[query_crop(q,duration) for q in protocol['queries'] if q['evaluate']]
        with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers,initializer=init_query_model,initargs=(str(a.model),)) as pool:
            for i,(_,_) in enumerate(pool.map(cache_query,[(q,str(a.cache)) for q in queries])):
                if (i+1)%200==0:print(f'Cached {duration:g}s queries {i+1}/{len(queries)}',flush=True)
        for kind,factor,stride in settings:
            if kind=='exact' and factor==1 and duration!=5:continue
            out=a.output/f'acr_{kind}_d{factor}_qs{stride}_{duration:g}s.json'
            if out.exists():
                prior=json.loads(out.read_text())
                if prior['protocol_sha256']!=hashlib.sha256(a.protocol.read_bytes()).hexdigest():raise ValueError('Existing output uses a different protocol')
                print(f'Existing completed {out.name}',flush=True);continue
            print(f'Building {kind} factor{factor} querystride{stride} duration{duration:g}',flush=True)
            index,stats=build_index(gallery,model,a.cache,factor,kind,a.frozen_ivf)
            closed=protocol['protocol']['closed_set']
            if closed:
                rows=evaluate_queries(queries,index,model,a.cache,stride);threshold=None
            else:
                cal=[q for q in queries if q['role'].startswith('calibration')]
                test=[q for q in queries if q['role'].startswith('test')]
                rows=evaluate_queries(cal,index,model,a.cache,stride)
                threshold=threshold_from_unknown([r['score'] for r in rows if r['role']=='calibration_unknown'],target=.01)
                cal_state={'threshold':threshold,'target':.01,'only_role':'calibration_unknown',
                    'calibration_unknown_queries':sum(r['role']=='calibration_unknown' for r in rows),
                    'heldout_queries_scored_before_gate':0,'threshold_model_or_setting_selected_from_test':False}
                (a.output/f'gate_{kind}_d{factor}_qs{stride}_{duration:g}s.json').write_text(json.dumps(cal_state,indent=2)+'\n')
                rows+=evaluate_queries(test,index,model,a.cache,stride)
                for row in rows:row['accepted']=bool(row['score']>=threshold and row['reference_id'] is not None)
            result={'method':'ACR-PCA frozen FMA transfer','dataset':protocol['dataset'],'dataset_doi':protocol['dataset_doi'],
                'closed_set':closed,'duration_s_requested':duration,'factor':factor,'config':asdict(model.config),
                'matcher':{'top_k':5,'tolerance_sec':.12,'exact':kind=='exact','query_stride':stride,
                    'ivf_nlist':512 if kind=='ivf' else None,'nprobe':16 if kind=='ivf' else None,'threads':a.threads,
                    'score':'vote_fraction/(1+mean_squared_L2)'},'frozen_threshold':threshold,
                'threshold_calibration':None if closed else cal_state,'stats':stats,'predictions':rows,
                'protocol_sha256':hashlib.sha256(a.protocol.read_bytes()).hexdigest(),
                'frozen_model_sha256':model_hash,'frozen_model_fit_source_count':len(model.fit_content_ids),
                'frozen_ivf_source_index_sha256':hashlib.sha256(a.frozen_ivf.read_bytes()).hexdigest() if kind=='ivf' else None,
                'representation_refit_on_sdrr':False,'settings_selected_from_sdrr_test':False,
                'protocol':protocol['protocol'],'provenance':protocol['provenance'],
                'fma_overlap_audit_sha256':hashlib.sha256(a.overlap_audit.read_bytes()).hexdigest(),
                'environment':{'platform':platform.platform(),'numpy':np.__version__,'faiss':faiss.__version__,'audio_decoder':audio_decoder_info()},
                'timing_caveat':'Concurrent experiments; these runs make no encoder/matcher latency claim.'}
            enrich_result(result,overlap)
            out.write_text(json.dumps(result,indent=2)+'\n');print(f'Saved {out.name}: {result["summary"]}',flush=True)
            del index;gc.collect()

if __name__=='__main__':main()
