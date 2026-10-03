"""Serial common-cohort CPU profile of encoder→frozen IVF/query-stride4.

Run only in the coordinated quiet window. All physical5s PCM arrays are loaded
before timing. End-to-end measurements are actual complete pipeline calls,
never a sum of separately aggregated encoder/matcher medians.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import platform
import resource
import subprocess
import time
import numpy as np
from acr_fp import Fingerprinter
from study_affine_calibration import load_frozen_index
from study_query_ivf import thin_query


def measured_pipeline(model,index,audio,sample_rate):
    start_wall=time.perf_counter_ns();start_cpu=time.process_time_ns()
    x,t=model.extract(audio,sample_rate)
    encoder_wall=time.perf_counter_ns();encoder_cpu=time.process_time_ns()
    xx,tt=thin_query(x,t,4,model.config)
    hits=index.match(xx,tt,top_k=5,tolerance_sec=.12,limit=1)
    end_wall=time.perf_counter_ns();end_cpu=time.process_time_ns()
    return {"encoder_wall_ms":(encoder_wall-start_wall)/1e6,
        "matcher_wall_ms":(end_wall-encoder_wall)/1e6,
        "end_to_end_wall_ms":(end_wall-start_wall)/1e6,
        "encoder_cpu_ms":(encoder_cpu-start_cpu)/1e6,
        "matcher_cpu_ms":(end_cpu-encoder_cpu)/1e6,
        "end_to_end_cpu_ms":(end_cpu-start_cpu)/1e6,
        "encoder_fingerprints":len(x),"searched_fingerprints":len(xx),
        "reference_id":hits[0]["content_id"] if hits else None,
        "score":hits[0]["score"] if hits else 0.}


def distribution(values):
    x=np.asarray(values,dtype=float)
    return {"median":float(np.median(x)),"p95":float(np.percentile(x,95)),
            "min":float(np.min(x)),"max":float(np.max(x))}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--inputs",type=Path,required=True)
    p.add_argument("--protocol",type=Path,required=True)
    p.add_argument("--model",type=Path,required=True)
    p.add_argument("--cache",type=Path,required=True)
    p.add_argument("--index",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--threads",type=int,default=4)
    p.add_argument("--warmups",type=int,default=2)
    p.add_argument("--repeats",type=int,default=5)
    a=p.parse_args()
    import faiss
    faiss.omp_set_num_threads(a.threads)
    inputs=json.loads(a.inputs.read_text())
    audio_path=Path(inputs["audio_path"])
    if hashlib.sha256(audio_path.read_bytes()).hexdigest()!=inputs["audio_sha256"]:
        raise ValueError("Common PCM input hash mismatch")
    audio=np.load(audio_path,allow_pickle=False)
    if audio.shape!=(50,40000) or audio.dtype!=np.float32 or inputs["sample_rate"]!=8000:
        raise ValueError("Common physical cohort is not50×5s/8k float32")
    protocol=json.loads(a.protocol.read_text())
    start_wall=time.perf_counter_ns();start_cpu=time.process_time_ns()
    model=Fingerprinter.load(a.model)
    model_setup={"wall_ms":(time.perf_counter_ns()-start_wall)/1e6,
                 "cpu_ms":(time.process_time_ns()-start_cpu)/1e6}
    start_wall=time.perf_counter_ns();start_cpu=time.process_time_ns()
    index=load_frozen_index(protocol,a.cache,model,a.index)
    index_setup={"wall_ms":(time.perf_counter_ns()-start_wall)/1e6,
                 "cpu_ms":(time.process_time_ns()-start_cpu)/1e6}
    first_call=measured_pipeline(model,index,audio[0],8000)
    rows=[]
    metrics=("encoder_wall_ms","matcher_wall_ms","end_to_end_wall_ms",
             "encoder_cpu_ms","matcher_cpu_ms","end_to_end_cpu_ms")
    for i,y in enumerate(audio):
        for _ in range(a.warmups):
            measured_pipeline(model,index,y,8000)
        repetitions=[measured_pipeline(model,index,y,8000) for _ in range(a.repeats)]
        predicted={r["reference_id"] for r in repetitions}
        if len(predicted)!=1:
            raise ValueError("Repeated fixed-input timing changed candidate identity")
        for repeat in repetitions:
            if abs(repeat["end_to_end_wall_ms"]-repeat["encoder_wall_ms"]-repeat["matcher_wall_ms"])>1e-9:
                raise ValueError("Pipeline timing stages do not add exactly")
        rows.append({"query_id":inputs["queries"][i]["query_id"],
            "source_id":inputs["queries"][i]["source_id"],"duration_s":5.,
            "repetitions":repetitions,"per_query_medians":{key:float(np.median([r[key] for r in repetitions])) for key in metrics}})
        if (i+1)%10==0:
            print(f"Common PCA/IVF timed {i+1}/50",flush=True)
    summaries={key:distribution([r["per_query_medians"][key] for r in rows]) for key in metrics}
    all_repeat={key:distribution([r[key] for row in rows for r in row["repetitions"]]) for key in metrics}
    cpu_brand=subprocess.run(["sysctl","-n","machdep.cpu.brand_string"],capture_output=True,text=True).stdout.strip()
    result={"method":"ACR-PCA32 + IVF512 factor8/query-stride4/nprobe16",
        "common_input_sha256":inputs["audio_sha256"],"input_manifest_sha256":hashlib.sha256(a.inputs.read_bytes()).hexdigest(),
        "input_count":50,"input_duration_s":5.,"sample_rate":8000,
        "model_setup":model_setup,"index_setup":index_setup,"first_5s_pipeline_call":first_call,
        "warmups_per_query":a.warmups,"repeats_per_query":a.repeats,
        "per_query_median_distributions":summaries,"all_repeat_distributions":all_repeat,
        "configuration":{"reference_factor":8,"query_stride":4,"nprobe":16,"nlist":512,"faiss_threads":a.threads,"top_k":5,"tolerance_sec":.12},
        "scope":"Encoder includes preprocessing, STFT, mel, mean/delta normalization, silence gate and PCA. Matcher includes original-grid query thinning, ANN and temporal aggregation. End-to-end is a single timed encoder→matcher call.",
        "exclusions":"File I/O/decoding, model/index setup, network and display are outside per-query timing.",
        "quiet_window":"Coordinated serial benchmark slot; heavy exact process stopped and other encoder profiles idle.",
        "environment":{"platform":platform.platform(),"machine":platform.machine(),"cpu_brand":cpu_brand,"numpy":np.__version__,"faiss":faiss.__version__},
        "peak_process_rss_platform_units":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "memory_note":"Process RSS includes the server index/libraries; not device encoder RAM.",
        "records":rows}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="records"},indent=2),flush=True)


if __name__=="__main__":
    main()
