"""Rate conversion for the same common50 arrays, outside all encoder clocks."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.signal import resample_poly


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--common", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    args = p.parse_args()
    metadata = json.loads((args.common / "inputs.json").read_text())
    audio_path = args.common / "audio.npy"
    if hashlib.sha256(audio_path.read_bytes()).hexdigest() != metadata["audio_sha256"]:
        raise ValueError("Common cohort audio checksum mismatch")
    audio = np.load(audio_path)
    if audio.shape != (50, 40000):
        raise ValueError("Expected exactly common50 x5s x8k")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sound = []
    for i, (wave, query) in enumerate(zip(audio, metadata["queries"])):
        signal = resample_poly(wave, 689, 1000).astype("<f4")  #8000→5512.
        path = args.output_dir / f"sound_{i:02d}.f32"
        path.write_bytes(signal.tobytes())
        sound.append({"query_id": query["query_id"], "source_id": query["source_id"],
                      "pcm_path": str(path.resolve()), "samples": len(signal), "duration_s": len(signal)/5512})
    landmark = resample_poly(audio, 441, 320, axis=1).astype("float32")  #8000→11025.
    # Native audfprint reader emits normalized PCM16. Quantization belongs to
    # input preparation, not the measured landmark encoder/matcher.
    landmark = np.clip(np.rint(landmark * 32768), -32768, 32767).astype("int16").astype("float32") / 32768
    np.save(args.output_dir / "audfprint.npy", landmark)
    out = {"common_input_sha256": metadata["audio_sha256"], "common_selection_seed": metadata["selection_seed"],
           "common_queries": metadata["queries"], "input_count": 50,
           "conversion": "SciPy resample_poly outside encoder clocks; native audfprint PCM16 quantization also outside",
           "soundfingerprinting_sample_rate": 5512, "soundfingerprinting_inputs": sound,
           "audfprint_sample_rate": 11025, "audfprint_audio": str((args.output_dir / "audfprint.npy").resolve())}
    (args.output_dir / "inputs.json").write_text(json.dumps(out, indent=2) + "\n")
    print(args.output_dir / "inputs.json")


if __name__ == "__main__":
    main()
