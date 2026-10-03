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

SD-RR v1.0.0 is the [Song Describer Real Re-recording release by Jiheng Li](https://doi.org/10.5281/zenodo.22169646), accompanying [POLARIS](https://github.com/JihengLi/POLARIS). It provides real laptop-speaker-to-phone recordings. Its authored metadata and documentation are CC BY-SA 4.0; embedded Song Describer metadata retain upstream terms. Audio licenses are per work, including CC BY, CC BY-SA, CC BY-NC, CC BY-NC-SA and Free Art License. Preserve the release attribution.csv, LICENSES.md, source URLs and exact versions. The code MIT license does not cover those data. The export includes metadata, derived measurements and small fitted states, but no audio, PCM, private TV recordings or rights-clearance claim.

The corrected canonical PeakNetFP adapter uses track-local physical time for catalog rounding. Obsolete catalog-rounding compatibility records are retained only in local audit storage and are not exported as primary results. Final CPU comparisons use the corrected Peak profile and actual compact-ACR/sparse-NMFP encoder calls; older timing tables are not primary V2 evidence.
