"""Fresh-process enrollment/memory experiment with unchanged ACR decisions.

Both variants use identical source roles, PCA and frozen IVF centroids. Gallery
prefixes retain every calibration-known source and add original known-test
reference sources by a fixed hash. Query timing uses the fixed common50 cohort.
No unknown source is enrolled and no test outcome selects a setting.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import numpy as np
import psutil

WORKSPACE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(WORKSPACE/"outputs/acr_repro"))
from acr_fp import Fingerprinter,ExactIndex
from run_ablations import ref_path
from study_query_ivf import thin_query
from compact_streaming_index import streaming_enroll


def other_cpu_snapshot(excluded):
    rows={}
    for process in psutil.process_iter(['pid','name','cpu_times']):
        try:
            if process.pid in excluded:continue
            c=process.info['cpu_times']
            if c is None:continue
            rows[process.pid]=(process.info['name'],float(c.user+c.system))
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    return rows


def cpu_activity(before,after,wall_s):
    rows=[]
    for pid,(name,value) in after.items():
        if pid in before:
            used=value-before[pid][1]
            if used>.05:rows.append({'pid':pid,'name':name,'cpu_s':used,'single_core_percent':100*used/wall_s})
    return sorted(rows,key=lambda r:-r['cpu_s'])


def rss():
    import psutil
    return int(psutil.Process().memory_info().rss)


def peak_rss():
    value=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if platform.system()=="Darwin" else value*1024)


def timed_queries(index,model,audio,inputs,repeats=5,warmups=2):
    # Query encoding is fixed and excluded from matcher timing.
    features=[model.extract(y,8000) for y in audio]
    rows=[]
    for q,(x,t) in zip(inputs["queries"],features):
        x,t=thin_query(x,t,4,model.config)
        for _ in range(warmups):index.match(x,t,top_k=5,tolerance_sec=.12,limit=1)
        wall,cpu,hits=[],[],[]
        for _ in range(repeats):
            w=time.perf_counter_ns();c=time.process_time_ns()
            hit=index.match(x,t,top_k=5,tolerance_sec=.12,limit=1)
            wall.append((time.perf_counter_ns()-w)/1e6);cpu.append((time.process_time_ns()-c)/1e6)
            hits.append(hit[0] if hit else {})
        if len({h.get("content_id") for h in hits})>1:raise ValueError("Timing changed repeated candidate")
        rows.append({"query_id":q["query_id"],"source_id":q["source_id"],"searched_fingerprints":len(x),
            "wall_ms":wall,"cpu_ms":cpu,"median_wall_ms":float(np.median(wall)),
            "median_cpu_ms":float(np.median(cpu)),"hit":hits[0]})
    return rows


def worker(a):
    import faiss
    faiss.omp_set_num_threads(a.threads)
    plan=json.loads(a.plan.read_text())
    model=Fingerprinter.load(plan["model_path"])
    gallery=plan["galleries"][str(a.size)]
    before=rss();start=time.perf_counter_ns();start_cpu=time.process_time_ns()
    ivf=faiss.read_index(plan["empty_ivf_path"]);ivf.nprobe=16
    duration_total=0.
    for r in gallery:
        with np.load(ref_path(plan["cache"],r,model.config),allow_pickle=False) as z:
            duration_total+=float(z["duration_s"])
    if a.variant=="baseline":
        entries=[]
        for reference in gallery:
            with np.load(ref_path(plan["cache"],reference,model.config),allow_pickle=False) as z:
                entries.append((reference["reference_id"],model.transform(z["features"]),z["times"].copy()))
        index=ExactIndex.from_entries(entries,factor=8,
            grid_step_sec=model.config.hop_length/model.config.sample_rate,
            grid_origin_sec=model.config.first_center_seconds)
        ivf.add(index._search_vectors);index._faiss=ivf
        digest=hashlib.sha256(memoryview(index.vectors).cast("B")).hexdigest()
        metadata_bytes=index.content_codes.nbytes+index.times.nbytes+sum(len(x.encode()) for x in index.content_ids)
        duplicate_vector_bytes=index.vectors.nbytes
        norm_bytes=index._norms.nbytes
        del entries
    else:
        index,digest=streaming_enroll(gallery,model,plan["cache"],ivf,factor=8)
        metadata_bytes=index.metadata_bytes();duplicate_vector_bytes=norm_bytes=0
    enrollment_wall_s=(time.perf_counter_ns()-start)/1e9
    enrollment_cpu_s=(time.process_time_ns()-start_cpu)/1e9
    enrollment_peak=peak_rss()
    gc.collect();time.sleep(.05)
    steady=rss()
    serialized_bytes=int(faiss.serialize_index(ivf).nbytes)
    stem=f"{a.variant}_{a.size}_r{a.replicate}"
    index_output=a.output/f"{stem}.index"
    save_start=time.perf_counter()
    faiss.write_index(ivf,str(index_output))
    serialization_wall_s=time.perf_counter()-save_start
    if a.replicate==0:
        if a.variant=='compact':index.save_metadata(a.output/f'{a.variant}_{a.size}_metadata.npz')
        else:np.savez_compressed(a.output/f'{a.variant}_{a.size}_metadata.npz',content_codes=index.content_codes,times=index.times,content_ids=np.asarray(index.content_ids))
    else:index_output.unlink()
    # Final steady query timing uses the same PCM and source cohort.
    inputs=json.loads(Path(plan["inputs_manifest"]).read_text())
    if hashlib.sha256(Path(inputs["audio_path"]).read_bytes()).hexdigest()!=inputs["audio_sha256"]:
        raise ValueError("Common cohort audio hash differs")
    audio=np.load(inputs["audio_path"],allow_pickle=False)
    rows=timed_queries(index,model,audio,inputs)
    med=[r["median_wall_ms"] for r in rows]
    result={"variant":a.variant,"replicate":a.replicate,"gallery_sources":len(gallery),"reference_audio_s":duration_total,
        "reference_fingerprints":ivf.ntotal,"descriptor_sha256":digest,
        "rss_before_enrollment_bytes":before,"rss_after_enrollment_gc_bytes":steady,
        "rss_enrollment_delta_bytes":steady-before,"process_peak_enrollment_rss_bytes":enrollment_peak,
        "peak_minus_before_bytes":enrollment_peak-before,
        "enrollment_wall_s":enrollment_wall_s,"enrollment_cpu_s":enrollment_cpu_s,
        "serialization_wall_s":serialization_wall_s,"serialized_index_bytes":serialized_bytes,
        "retained_metadata_bytes":metadata_bytes,"retained_python_vector_bytes":duplicate_vector_bytes,
        "retained_norm_cache_bytes":norm_bytes,
        "metadata_accounting":"Per-row arrays, track boundaries when present, and UTF-8 content-ID lookup bytes; Python container overhead excluded.",
        "explicit_reference_bytes":serialized_bytes+metadata_bytes+duplicate_vector_bytes+norm_bytes,
        "median_match_wall_ms":float(np.median(med)),"p95_match_wall_ms":float(np.percentile(med,95)),
        "queries":rows,"configuration":{"factor":8,"query_stride":4,"nlist":512,"nprobe":16,"threads":a.threads},
        "environment":{"platform":platform.platform(),"numpy":np.__version__,"faiss":faiss.__version__},
        "enrollment_scope":"Cached feature reads, PCA projection, metadata and index population; frozen centroid training excluded.",
        "query_scope":"Original-grid query thinning done before timing; matching includes ANN and the unchanged temporal vote rule."}
    out=a.output/f"{stem}.json";out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="queries"}),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--protocol",type=Path,default=WORKSPACE/"work/benchmarks/hard_medium_protocol.json")
    p.add_argument("--model",type=Path,default=WORKSPACE/"work/benchmarks/acr_medium/pca_model.npz")
    p.add_argument("--cache",type=Path,default=WORKSPACE/"work/benchmarks/acr_cache_ffmpeg")
    p.add_argument("--frozen-index",type=Path,default=WORKSPACE/"work/benchmarks/acr_ivf_study/ivf_factor8.index")
    p.add_argument("--inputs",type=Path,default=WORKSPACE/"work/profiling/common_inputs/inputs.json")
    p.add_argument("--output",type=Path,default=WORKSPACE/"work/benchmarks/compact_store")
    p.add_argument("--threads",type=int,default=4)
    p.add_argument("--worker",action="store_true")
    p.add_argument("--plan",type=Path)
    p.add_argument("--variant",choices=["baseline","compact"])
    p.add_argument("--size",type=int)
    p.add_argument("--replicate",type=int,default=0)
    p.add_argument("--replicates",type=int,default=3)
    p.add_argument("--sizes",default="141,256,512,659")
    p.add_argument("--prepare-only",action="store_true")
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if a.worker:return worker(a)
    import faiss
    protocol=json.loads(a.protocol.read_text())
    original=[r for r in protocol["references"] if r["role"] in ("calibration_known","test_known")]
    mandatory={r["reference_id"] for r in original if r["role"]=="calibration_known"}
    extras=[r for r in original if r["reference_id"] not in mandatory]
    extras.sort(key=lambda r:hashlib.sha256(("acr-memory-scaling-v1:"+r["reference_id"]).encode()).hexdigest())
    sizes=[int(v) for v in a.sizes.split(",")]
    if any(size<len(mandatory) or size>len(original) for size in sizes):raise ValueError("Invalid role-safe gallery size")
    galleries={}
    for size in sizes:
        ids=mandatory|{r["reference_id"] for r in extras[:size-len(mandatory)]}
        galleries[str(size)]=[r for r in original if r["reference_id"] in ids]
    full=faiss.read_index(str(a.frozen_index));full.reset();full.set_direct_map_type(faiss.DirectMap.NoMap)
    template=a.output/"frozen_empty_ivf512.index"
    plan={"protocol_sha256":hashlib.sha256(a.protocol.read_bytes()).hexdigest(),
        "model_path":str(a.model),"model_sha256":hashlib.sha256(a.model.read_bytes()).hexdigest(),
        "frozen_index_sha256":hashlib.sha256(a.frozen_index.read_bytes()).hexdigest(),
        "empty_ivf_path":str(template),"cache":str(a.cache),"inputs_manifest":str(a.inputs),
        "sizes":sizes,"replicates":a.replicates,"galleries":galleries,"selection_seed":"acr-memory-scaling-v1",
        "threads":a.threads,"matcher_warmups":2,"matcher_repeats":5,
        "input_manifest_sha256":hashlib.sha256(a.inputs.read_bytes()).hexdigest(),
        "code_sha256":{name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("run_streaming_index_study.py","compact_streaming_index.py","acr_fp.py")},
        "all_calibration_known_sources_retained":True,"unknown_sources_enrolled":False,
        "test_outcomes_used_to_choose_settings":False,"feature_and_centroid_settings_frozen":True}
    plan_path=a.output/"plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text())!=plan:
        raise ValueError("Existing experiment plan differs; use a fresh output directory")
    faiss.write_index(full,str(template));del full;gc.collect()
    plan_path.write_text(json.dumps(plan,indent=2)+"\n")
    if a.prepare_only:return
    for size in sizes:
        for repeat in range(a.replicates):
            order=('baseline','compact') if repeat%2==0 else ('compact','baseline')
            for variant in order:
                stem=f'{variant}_{size}_r{repeat}';out=a.output/f'{stem}.json'
                if out.exists():continue
                command=[sys.executable,str(Path(__file__).resolve()),"--worker","--plan",str(plan_path),
                    "--size",str(size),"--variant",variant,"--output",str(a.output),"--threads",str(a.threads),
                    '--replicate',str(repeat)]
                with (a.output/f'{stem}.log').open('w') as log:
                    child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
                    excluded={child.pid,psutil.Process().pid}
                    before=other_cpu_snapshot(excluded);start=time.perf_counter()
                    status=child.wait()
                    elapsed=time.perf_counter()-start;after=other_cpu_snapshot(excluded)
                    if status:raise subprocess.CalledProcessError(status,command)
                result=json.loads(out.read_text())
                result['other_process_cpu_activity']=cpu_activity(before,after,elapsed)
                result['worker_total_wall_s']=elapsed
                out.write_text(json.dumps(result,indent=2)+'\n')
                print(f'Completed fresh-process {variant} gallery{size} repeat{repeat}',flush=True)
    summaries=[]
    for size in sizes:
        left=json.loads((a.output/f"baseline_{size}_r0.json").read_text());right=json.loads((a.output/f"compact_{size}_r0.json").read_text())
        comparisons=[]
        for l,r in zip(left["queries"],right["queries"]):
            if l["query_id"]!=r["query_id"]:raise ValueError("Cost cohorts differ")
            lh,rh=l["hit"],r["hit"]
            comparisons.append({"query_id":l["query_id"],"same_id":lh.get("content_id")==rh.get("content_id"),
                "same_votes":lh.get("votes")==rh.get("votes"),"score_abs_difference":abs(lh.get("score",0)-rh.get("score",0)),
                "offset_abs_difference":abs(lh.get("offset_sec",0)-rh.get("offset_sec",0))})
        summaries.append({"gallery_sources":size,"baseline":{k:v for k,v in left.items() if k!="queries"},
            "compact":{k:v for k,v in right.items() if k!="queries"},
            "descriptor_bytes_identical":left["descriptor_sha256"]==right["descriptor_sha256"],
            "same_candidate_ids":sum(r["same_id"] for r in comparisons),"same_vote_counts":sum(r["same_votes"] for r in comparisons),
            "query_count":len(comparisons),"max_score_abs_difference":max(r["score_abs_difference"] for r in comparisons),
            "max_offset_abs_difference":max(r["offset_abs_difference"] for r in comparisons),"parity_details":comparisons})
        for variant in ('baseline','compact'):
            replicas=[json.loads((a.output/f'{variant}_{size}_r{r}.json').read_text()) for r in range(a.replicates)]
            if len({r['descriptor_sha256'] for r in replicas})!=1:raise ValueError('Replicate descriptors differ')
            fields=['rss_after_enrollment_gc_bytes','rss_enrollment_delta_bytes','process_peak_enrollment_rss_bytes',
                    'enrollment_wall_s','enrollment_cpu_s','median_match_wall_ms','p95_match_wall_ms']
            summaries[-1][variant]['replicate_aggregate']={field:{'values':[r[field] for r in replicas],
                'median':float(np.median([r[field] for r in replicas]))} for field in fields}
    (a.output/"summary.json").write_text(json.dumps({"plan":plan,"rows":summaries},indent=2)+"\n")
    print("Streaming enrollment/memory study complete",flush=True)


if __name__=="__main__":main()
