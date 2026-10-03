"""Common50 warm encoder+matching timing with rate conversion/file I/O excluded."""
import argparse
import json
from pathlib import Path
import platform
import time
import numpy as np
from audfprint_baseline import AudfprintBaseline


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--protocol", required=True, type=Path)
    p.add_argument("--inputs", required=True, type=Path)
    p.add_argument("--upstream", required=True, type=Path)
    p.add_argument("--cache", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    protocol = json.loads(a.protocol.read_text())
    common = json.loads(a.inputs.read_text())
    waves = np.load(common["audfprint_audio"])
    refs = [r for r in protocol["references"] if r["role"] in protocol["protocol"]["gallery_roles"]]
    setup = time.perf_counter()
    baseline = AudfprintBaseline(a.upstream, a.cache, min_count=1)
    baseline.fit(refs, workers=1)
    setup_s = time.perf_counter() - setup
    import audio_read
    reader = audio_read.audio_read
    rows = []
    try:
        for wave, query in zip(waves, common["common_queries"]):
            # Native implementation unchanged; decoded fixture is supplied at
            # the normal audio_read boundary without file reads/process spawn.
            audio_read.audio_read = lambda *args, **kwargs: (wave, 11025)
            encoder, matching, pipeline, cpu = [], [], [], []
            first = None
            for repeat in range(-2, 5):
                cpu_start = time.process_time()
                start = time.perf_counter()
                hashes = baseline.analyzer.wavfile2hashes("common_fixture.wav")
                split = time.perf_counter()
                results = baseline.matcher.match_hashes(baseline.table, hashes)
                stop = time.perf_counter()
                cpu_stop = time.process_time()
                if first is None:
                    first = stop - start
                if repeat >= 0:
                    encoder.append(split - start); matching.append(stop - split)
                    pipeline.append(stop - start); cpu.append(cpu_stop - cpu_start)
            rows.append({"query_id": query["query_id"], "source_id": query["source_id"],
                         "encoder_s": encoder, "matching_s": matching, "pipeline_s": pipeline,
                         "process_cpu_s": cpu, "first_actual_shape_pipeline_s": first,
                         "fingerprint_count": len(hashes), "sample_rate": 11025, "sample_count": len(wave)})
    finally:
        audio_read.audio_read = reader
    out = {"method": "audfprint candidate floor1 warmed CPU", "config": baseline.config,
           "common_input_sha256": common["common_input_sha256"], "queries": rows,
           "repeats": 5, "warmups": 2, "index_setup": baseline.stats, "total_setup_wall_s": setup_s,
           "environment": {"platform": platform.platform(), "numpy": np.__version__},
           "included": "Native four-shift landmark encoding and min-count1 matching",
           "excluded": "File I/O, rate conversion, PCM16 quantization, model/index setup",
           "note": "Pipeline time measured within each query, not sum of separate medians; encoder inputs supplied at upstream reader boundary"}
    a.output.write_text(json.dumps(out, indent=2) + "\n")
    print(a.output)


if __name__ == "__main__":
    main()
