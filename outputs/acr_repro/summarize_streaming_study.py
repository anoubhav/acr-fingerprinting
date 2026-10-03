"""Emit a path-free publication summary and vector figure from frozen measurements."""
from pathlib import Path
import argparse
import json
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--experiment',type=Path,default=ROOT/'work/benchmarks/compact_store')
OUT=p.parse_args().experiment
x=json.loads((OUT/'summary.json').read_text());parity=json.loads((OUT/'full835_parity.json').read_text())
rows=[]
for row in x['rows']:
    output={'gallery_sources':row['gallery_sources'],'audio_hours':row['baseline']['reference_audio_s']/3600,
        'reference_fingerprints':row['baseline']['reference_fingerprints'],
        'reference_descriptor_sha256':row['baseline']['descriptor_sha256'],
        'descriptor_bytes_identical':row['descriptor_bytes_identical'],'calibration_query_count':row['query_count']}
    for kind in ['baseline','compact']:
        original=row[kind];a=original['replicate_aggregate']
        output[kind]={'explicit_reference_bytes':original['explicit_reference_bytes'],
            'serialized_index_bytes':original['serialized_index_bytes'],
            'retained_metadata_bytes':original['retained_metadata_bytes'],
            'retained_python_vector_bytes':original['retained_python_vector_bytes'],
            'retained_norm_cache_bytes':original['retained_norm_cache_bytes'],
            'measurements':a}
    rows.append(output)
code_paths=[ROOT/'outputs/acr_repro/acr_fp.py',ROOT/'outputs/acr_repro/run_ablations.py',
    ROOT/'outputs/acr_repro/study_query_ivf.py',ROOT/'outputs/acr_repro/compact_streaming_index.py',
    ROOT/'outputs/acr_repro/run_streaming_index_study.py',
    ROOT/'outputs/acr_repro/check_compact_full_parity.py']
public={'experiment':'Frozen ACR compact-store and streaming-enrollment conformance/scaling',
    'settings':{'reference_factor':8,'query_stride':4,'ivf_nlist':512,'nprobe':16,'faiss_threads':4,
        'gallery_sizes':[141,256,512,659],'fresh_process_replicates':3,'order':'baseline/compact,compact/baseline,baseline/compact',
        'query_count':50,'query_duration_s':5,'matcher_warmups_per_query':2,'matcher_repeats_per_query':5,
        'timing_aggregation':'Per-query repeat median, then cohort median/p95; median across three fresh processes.'},
    'cohort':{'selection_seed':x['plan']['selection_seed'],'all_calibration_known_sources_retained':True,
        'unknown_sources_enrolled':False,'test_outcomes_used_to_choose_settings':False,
        'galleries':{n:[{'reference_id':r['reference_id'],'role':r['role']} for r in refs] for n,refs in x['plan']['galleries'].items()}},
    'provenance':{'protocol_sha256':x['plan']['protocol_sha256'],'model_sha256':x['plan']['model_sha256'],
        'frozen_index_sha256':x['plan']['frozen_index_sha256'],
        'code_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in code_paths}},
    'measurements':rows,'full_cohort_conformance':{k:v for k,v in parity.items() if k!='rows'},
    'scope':{'enrollment':'Cached raw feature reads, PCA projection, metadata, IVF population. Frozen centroid training and audio/frontend processing excluded.',
        'rss':'macOS fresh-process RSS; peak from getrusage is bytes. Post-enrollment after Python GC; allocator-retained workspace remains part of process RSS.',
        'logical_state':'Serialized IVF size plus explicitly retained reference arrays. This is logical explicit state, not a claim about FAISS list capacity or total allocated memory.',
        'comparison':'Within this frozen ACR implementation only; not a compactness frontier against optimized neural/hash baselines.',
        'latency':'ANN search plus unchanged temporal voting. Query encoding and thinning excluded. All foreground benchmark jobs held; stable desktop/recording activity logged in worker JSONs.'}}
(OUT/'publication_summary.json').write_text(json.dumps(public,indent=2)+'\n')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,axes=plt.subplots(1,3,figsize=(8.9,2.6))
hours=[r['audio_hours'] for r in rows]
for method,label,color,marker in [('baseline','Materialized','#4A5B73','o'),('compact','Streaming + compact','#D17B33','s')]:
    rss=[r[method]['measurements']['rss_after_enrollment_gc_bytes']['median']/1e6 for r in rows]
    rsslo=[min(r[method]['measurements']['rss_after_enrollment_gc_bytes']['values'])/1e6 for r in rows]
    rsshi=[max(r[method]['measurements']['rss_after_enrollment_gc_bytes']['values'])/1e6 for r in rows]
    axes[0].errorbar(hours,rss,yerr=[np.subtract(rss,rsslo),np.subtract(rsshi,rss)],label=label,color=color,marker=marker,capsize=2)
    explicit=[r[method]['explicit_reference_bytes']/1e6 for r in rows]
    axes[1].plot(hours,explicit,color=color,marker=marker)
    latency=[r[method]['measurements']['median_match_wall_ms']['median'] for r in rows]
    axes[2].plot(hours,latency,color=color,marker=marker)
for ax in axes:
    ax.set_xlabel('Reference audio (hours)');ax.grid(alpha=.15);ax.set_xlim(7,44)
axes[0].set_ylabel('Post-enrollment process RSS (MB)')
axes[1].set_ylabel('Logical explicit reference state (MB)')
axes[2].set_ylabel('Median matching time (ms / 5 s query)')
axes[0].legend(frameon=False,fontsize=7,loc='upper left')
fig.tight_layout();fig.savefig(OUT/'streaming_store.pdf',bbox_inches='tight');fig.savefig(OUT/'streaming_store.png',dpi=200,bbox_inches='tight')
print('Path-free summary and measured scaling figure written.')
