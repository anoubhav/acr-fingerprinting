# Maintained public Haar/MinHash baseline

This bridge runs the unmodified `SoundFingerprinting` NuGet package **15.14.1**,
whose package metadata identifies source commit
`9f882368258c76ac1776e192817d8f60eaa6f22b` in
<https://github.com/AddictedCS/soundfingerprinting>. Its MIT license is separate
from audio licenses. The dependency lock records NuGet content hashes.

This is a current public library baseline. It is not an implementation of the
supplied paper's proprietary 72-byte MinHash predecessor, and new public results
must not be labeled as a replication of that predecessor's historical tables.

## Input and defaults

All files are decoded by the benchmark's FFmpeg to mono float32 at 5512 Hz.
Exact centered query bounds are unchanged. The public `AudioSamples` API receives
these samples after the library's own `AudioSamplesNormalizer`; platform-specific
audio services are bypassed. No query context or padding is added. Native FFT,
Haar, MinHash, indexing, path reconstruction, and confidence functions are intact.

The defaults are FFT 2048/hop 64, 32 logarithmic bands over 318–2000 Hz, a
32-by-128 spectral image, 200 top Haar wavelets, and 100 MinHashes packed into
25 four-byte LSH keys (100-byte hash payload per fingerprint). Enrollment uses
512-sample increments. Queries use the native incremental random stride
256–512 samples (exclusive upper bound) with public seed 20261002 for repeatability.
All degradation variants of a crop use the same seeded sampling pattern.

The nominal fingerprint time is `8192/5512 = 1.486 s`. Complete FFT support
requires `(128*64 + 2048 - 64)/5512 = 1.846 s`. A 1 s query produces no spectral
image under this configuration. Such durations are marked unsupported rather
than padded or treated as a general limitation of MinHash.

Spectral profiles remain enabled at enrollment, as in this release's defaults;
query bridging is the default `NoBridgingStrategy`. Playback-speed compensation
is zero. Ground-truth tempo and pitch annotations never configure the library.

## Calibration and matching

The native descriptor-pair LSH threshold is four votes. An additional sensitivity
variant uses one vote. This threshold qualifies descriptor pairs before path
reconstruction; reducing it can cause dense associations and poor rankings. It
is not equivalent to removing a final scalar score floor. Both variants are
reported and the recommended native threshold remains the primary public-library
configuration.

The native `Audio.BestMatch` is used without label-based reranking. External
scores are its `ResultEntry.Confidence`; the native similarity score and coverage
are also preserved. Thresholds are chosen only on calibration unknown sources.
For calibration and acceptance, Confidence is rounded to 12 decimal places and
the original is retained as `raw_score`. Mathematically tied one-fingerprint
coverage values otherwise differ by about 1e-14 due to floating-point subtraction;
splitting those ties would give unstable rare-FPR operating points. Native
candidate ranking and labels remain unchanged.
The source offset is `TrackMatchStartsAt - QueryMatchStartsAt`. Repeated motifs
can correctly identify a track while failing the separate 2 s localization test.

Raw hash payload bytes, protobuf cache bytes, serialized index bytes, and managed
index memory delta are reported separately. Managed memory includes the library's
index object structure and is not a theoretical signature size. Timing includes
native normalization, feature/hash generation, and matching; FFmpeg decoding is
recorded separately. Prepared PCM keeps the exact decoder output and avoids slow
C# pipe transfer, with cold preparation time and worker count recorded.

## Build and run

Install a portable Microsoft SDK in `work/dotnet` (the measured SDK is 10.0.401).
The official archive's published SHA-512 was verified before extraction. Keep
`DOTNET_CLI_HOME`, NuGet package/HTTP caches, and temporary files under `work/`.
No global or home-directory installation is needed.

The measured macOS arm64 binary is available from
`https://builds.dotnet.microsoft.com/dotnet/Sdk/10.0.401/dotnet-sdk-10.0.401-osx-arm64.tar.gz`.
Its SHA-512 is
`69f64eb00dc045398755c440b152225d544301a345a146a16e86a56a0c52b7c94b2c331520e976dbb821f18d31930aafbd25bb85961e3517e0665414ce0cbcff`.
Download to `work/`, verify this digest, then extract to `work/dotnet/`. For other
systems, select the matching RID/10.0 SDK binary and published checksum from
Microsoft's official release metadata at
`https://builds.dotnet.microsoft.com/dotnet/release-metadata/10.0/releases.json`.

```sh
DOTNET_CLI_HOME="$PWD/work/dotnet_home" \
NUGET_PACKAGES="$PWD/work/nuget_packages" \
NUGET_HTTP_CACHE_PATH="$PWD/work/nuget_http" \
DOTNET_CLI_TELEMETRY_OPTOUT=1 \
DOTNET_GENERATE_ASPNET_CERTIFICATE=false \
work/dotnet/dotnet build outputs/acr_repro/soundfingerprinting_bridge/SoundFingerprintingBridge.csproj \
  --configuration Release -p:RestoreLockedMode=true --output "$PWD/work/soundfingerprinting_build" \
  -p:BaseIntermediateOutputPath="$PWD/work/soundfingerprinting_obj/"

work/venv/bin/python outputs/acr_repro/prepare_soundfingerprinting_pcm.py \
  --protocol work/benchmarks/hard_medium_protocol.json \
  --cache work/benchmarks/soundfingerprinting_pcm \
  --output work/benchmarks/soundfingerprinting_medium_pcm.json --durations 5 --workers 4

DOTNET_CLI_HOME="$PWD/work/dotnet_home" DOTNET_PROCESSOR_COUNT=4 \
work/dotnet/dotnet work/soundfingerprinting_build/SoundFingerprintingBridge.dll \
  --protocol work/benchmarks/hard_medium_protocol.json \
  --pcm-manifest work/benchmarks/soundfingerprinting_medium_pcm.json \
  --cache work/benchmarks/soundfingerprinting_cache \
  --native-output-dir work/benchmarks/soundfingerprinting_medium \
  --calibrated-output-dir work/benchmarks/soundfingerprinting_calibrated_medium --durations 5
```

The driver rejects PCM prepared for a different frozen protocol SHA. Reference
fingerprints are cached as the upstream protobuf type. Input audio, PCM caches,
SDKs, and model snapshots are not redistributed in the research deliverable.
