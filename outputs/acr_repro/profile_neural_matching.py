"""Profile native exact/IVF matching on the same warmed calibration inputs."""
import argparse,json,time
from pathlib import Path
import numpy as np
from neural_baseline import NMFPSequenceIndex,load_cached_embeddings

def run(args):
    import faiss
    faiss.omp_set_num_threads(4)
    encoder=json.loads(args.encoder_profile.read_text())
    cache=json.loads(args.reference_cache.read_text())
    index_class=NMFPSequenceIndex
    if args.method=='peaknet':
        from peaknet_baseline import PeakNetSequenceIndex
        index_class=PeakNetSequenceIndex
    index=index_class.from_files({rid:item['path'] for rid,item in cache['references'].items()})
    queries=[(q['query_id'],load_cached_embeddings(encoder['query_embedding_files'][q['query_id']])[0]) for q in encoder['inputs']['queries']]
    result={'method':encoder['method'],'inputs_sha256':encoder['inputs']['audio_sha256'],
            'input_count':len(queries),'threads':4,'top_k':20,'repeats':args.repeats,
            'reference_fingerprints':len(index.embeddings),'native_time_scaling':'estimated' if args.method=='peaknet' else 'unit'}
    for variant in ('exact','ivf'):
        build=(index.build_index() if variant=='exact' else index.build_cpu_ivf(1024,16))
        for _,query in queries[:3]:index.search(query,20)
        timing=[]
        for repeat in range(args.repeats):
            for qid,query in queries:
                tick=time.perf_counter();hit=index.search(query,20);elapsed=time.perf_counter()-tick
                timing.append({'repeat':repeat,'query_id':qid,'matching_s':elapsed,
                               'reference_id':hit['reference_id'],'score':hit['score']})
        values=np.array([r['matching_s'] for r in timing])
        result[variant]={'index_build_s':build,'nlist':1024 if variant=='ivf' else None,
                         'nprobe':16 if variant=='ivf' else None,'timing':timing,
                         'matching_s':{'median':float(np.median(values)),
                                       'p95':float(np.quantile(values,.95)),
                                       'p99':float(np.quantile(values,.99))},
                         'index_serialized_bytes':int(faiss.serialize_index(index._index).nbytes)}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print({k:result[k]['matching_s'] for k in ('exact','ivf')},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--method',choices=('nmfp','peaknet'),required=True);p.add_argument('--reference-cache',type=Path,required=True);p.add_argument('--encoder-profile',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--repeats',type=int,default=3)
    run(p.parse_args())
