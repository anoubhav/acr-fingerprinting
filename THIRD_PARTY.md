# Third-party sources

The MIT license applies to the new repository code. Upstream implementations and neural weights are downloaded separately. Their own license files and model/data terms remain authoritative.

The underlying non-neural method is credited to Anoubhav Agarwaal, Prabhat Kanaujia, Sartaki Sinha Roy, and Susmita Ghose: [Robust and lightweight audio fingerprint for Automatic Content Recognition](https://arxiv.org/abs/2305.09559), arXiv:2305.09559 (2023). Original method authorship and historical results are distinct from this repository's new code and public experiments.

| Component | Source | Observed source license |
|---|---|---|
| Audfprint | [dpwe/audfprint](https://github.com/dpwe/audfprint) | MIT |
| SoundFingerprinting | [AddictedCS/soundfingerprinting](https://github.com/AddictedCS/soundfingerprinting) | MIT |
| NMFP-Triplet | [raraz15/neural-music-fp](https://github.com/raraz15/neural-music-fp) | GPL-3.0 |
| PeakNetFP | [guillemcortes/peaknetfp](https://github.com/guillemcortes/peaknetfp) | CC BY-NC-SA 4.0 |
| Kapre 0.3.7 | [keunwoochoi/kapre](https://github.com/keunwoochoi/kapre) | MIT |

Pinned revisions, checkpoint archive digests, and execution changes are documented in [NMFP](outputs/acr_repro/NEURAL_BASELINE.md), [PeakNetFP](outputs/acr_repro/PEAKNET_BASELINE.md), and [SoundFingerprinting](outputs/acr_repro/SOUNDFINGERPRINTING_BASELINE.md).

PEX/FMA and LibriSpeech audio are acquired from their original providers. Audio is not redistributed in this repository. FMA track metadata includes per-track attribution and license information; source IDs in results do not replace those terms. The stored protocols and derived measurements identify the evaluated recordings and preserve provenance.

NumPy, SciPy, librosa, FAISS, TensorFlow, Essentia, .NET, and other installed dependencies have their own licenses. No dependency binaries or framework runtime are bundled.
