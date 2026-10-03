"""Measure warmed encoders on common predecoded PCM; separate cold setup/trace."""
import argparse,json,time,sys
from pathlib import Path
import numpy as np
from neural_baseline import NMFPExtractor,NMFPSequenceIndex

def run(args):
    manifest=json.loads(args.inputs.read_text());audio=np.load(manifest['audio_path'],allow_pickle=False)
    args.output.mkdir(parents=True,exist_ok=True)
    tick=time.perf_counter()
    if args.method=='peaknet':
        from peaknet_baseline import PeakNetExtractor
        extractor=PeakNetExtractor(batch_size=125,threads=4)
        extractor.query_mode=True
    else:
        extractor=NMFPExtractor(batch_size=256,threads=4)
    model_setup_s=time.perf_counter()-tick
    # Reuse the exact CPU FAISS binary used by the main environment. Import
    # NumPy/TF first so their compatible pinned versions remain selected.
    sys.path.append(str(Path(__file__).resolve().parents[2]/'work/venv/lib/python3.12/site-packages'))
    import faiss
    faiss.omp_set_num_threads(4)
    cache=json.loads(args.reference_cache.read_text())
    index_class=NMFPSequenceIndex
    if args.method=='peaknet':
        from peaknet_baseline import PeakNetSequenceIndex
        index_class=PeakNetSequenceIndex
    tick=time.perf_counter()
    index=index_class.from_files({rid:entry['path'] for rid,entry in cache['references'].items()})
    reference_cache_load_s=time.perf_counter()-tick
    tick=time.perf_counter();extractor.extract_array(audio[0]);first_actual_5s_s=time.perf_counter()-tick
    for _ in range(2):extractor.extract_array(audio[0])
    timing=[];files={};variants={}
    for variant in ('exact','ivf'):
        build_s=index.build_index() if variant=='exact' else index.build_cpu_ivf(1024,16)
        for _ in range(2):
            warm_embedding,_,_=extractor.extract_array(audio[0]);index.search(warm_embedding,20)
        rows=[]
        for repeat in range(args.repeats):
            for i,wave in enumerate(audio):
                pipeline_tick=time.perf_counter()
                tick=time.perf_counter();embedding,timestamps,parts=extractor.extract_array(wave)
                elapsed=time.perf_counter()-tick
                tick=time.perf_counter();hit=index.search(embedding,20)
                match_s=time.perf_counter()-tick
                pipeline_s=time.perf_counter()-pipeline_tick
                row={'variant':variant,'repeat':repeat,'query_id':manifest['queries'][i]['query_id'],
                     'encoder_s':elapsed,'matching_s':match_s,'pipeline_s':pipeline_s,
                     'reference_id':hit['reference_id'],'score':hit['score'],
                     'audio_s':5,'fingerprints':len(embedding),**parts}
                timing.append(row);rows.append(row)
                if repeat==0 and variant=='exact':
                    path=args.output/f'query_{i:03d}.npz'
                    np.savez(path,embeddings=embedding,times=timestamps,
                             metadata_json=json.dumps({'item':manifest['queries'][i],'timing':{'frontend_s':parts['frontend_s'],'forward_s':parts['forward_s']},'provenance':extractor.provenance}))
                    files[manifest['queries'][i]['query_id']]=str(path.resolve())
        variants[variant]={'index_build_s':build_s,
                          'nlist':1024 if variant=='ivf' else None,
                          'nprobe':16 if variant=='ivf' else None}
        for key in ('encoder_s','matching_s','pipeline_s'):
            samples=np.array([r[key] for r in rows])
            variants[variant][key]={'median':float(np.median(samples)),
                                  'p95':float(np.quantile(samples,.95)),
                                  'p99':float(np.quantile(samples,.99))}
    values=np.array([t['encoder_s'] for t in timing])
    weights_bytes=sum(int(np.prod(v.shape))*np.dtype(v.dtype.as_numpy_dtype).itemsize for v in extractor.model.variables)
    result={'method':extractor.provenance['method'],'inputs':manifest,'provenance':extractor.provenance,
            'model_parameter_count':extractor.model.count_params(),'model_weights_bytes':weights_bytes,
            'model_setup_s':model_setup_s,'first_actual_5s_call_s':first_actual_5s_s,
            'reference_cache_load_s':reference_cache_load_s,
            'faiss_version':faiss.__version__,'faiss_compile_options':faiss.get_compile_options(),
            'faiss_cpu_threads':4,'explicit_gpu_index':False,'pipeline_variants':variants,
            'cold_note':'Setup includes imports/restoration/constructor warm-up. First actual5s call separately includes any remaining shape trace; neither is part of steady timings.',
            'warm_policy':'Actual5s/9-window query shape warmed with3 calls before measurement; predecoded common PCM;4 CPU threads',
            'repeats':args.repeats,'timing':timing,'query_embedding_files':files,
            'encoder_s':{'median':float(np.median(values)),'p95':float(np.quantile(values,.95)),'p99':float(np.quantile(values,.99))},
            'encoder_ms_per_audio_s':float(np.median(values)*1000/5)}
    (args.output/'profile.json').write_text(json.dumps(result,indent=2)+'\n')
    print({k:result[k] for k in ('method','model_parameter_count','model_weights_bytes','model_setup_s','first_actual_5s_call_s','encoder_s','encoder_ms_per_audio_s','pipeline_variants')},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--method',choices=('nmfp','peaknet'),required=True);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--reference-cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--repeats',type=int,default=5)
    run(p.parse_args())
