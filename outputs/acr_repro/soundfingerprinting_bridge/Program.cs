using System.Diagnostics;
using System.Globalization;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using ProtoBuf;
using SoundFingerprinting;
using SoundFingerprinting.Audio;
using SoundFingerprinting.Builder;
using SoundFingerprinting.Configuration;
using SoundFingerprinting.Data;
using SoundFingerprinting.InMemory;
using SoundFingerprinting.Strides;

internal static class Program
{
    const int SampleRate = 5512;
    const int Seed = 20261002;
    const int MinimumSamples = 10176; //128*64+2048-64, pinned source defaults.
    const string PackageVersion = "15.14.1";
    const string UpstreamRevision = "9f882368258c76ac1776e192817d8f60eaa6f22b";
    static readonly DateTime Epoch = new(2000, 1, 1, 0, 0, 0, DateTimeKind.Utc);
    static readonly JsonSerializerOptions JsonOptions = new() { WriteIndented = true };

    static Dictionary<string, string> Options(string[] args)
    {
        var options = new Dictionary<string, string>();
        for (int i = 0; i < args.Length; i += 2)
            options.Add(args[i], args[i + 1]);
        return options;
    }

    static string Hash(string text) => Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(text))).ToLowerInvariant();
    static string Text(JsonElement item, string key) => item.GetProperty(key).GetString()!;

    static async Task<(float[] Samples, double DecodeSeconds, string Warning)> Decode(string path, double? start = null, double? duration = null, JsonElement? prepared = null)
    {
        var sw = Stopwatch.StartNew();
        if (prepared.HasValue)
        {
            byte[] raw = await File.ReadAllBytesAsync(Text(prepared.Value, "pcm_path"));
            float[] cached = new float[raw.Length / 4];
            Buffer.BlockCopy(raw, 0, cached, 0, raw.Length);
            if (cached.Length != prepared.Value.GetProperty("decoded_samples").GetInt32())
                throw new InvalidDataException("Prepared PCM sample count mismatch");
            return (cached, sw.Elapsed.TotalSeconds, Text(prepared.Value, "warning"));
        }
        var info = new ProcessStartInfo("ffmpeg") { RedirectStandardOutput = true, RedirectStandardError = true, UseShellExecute = false };
        foreach (string arg in new[] { "-v", "error", "-i", path }) info.ArgumentList.Add(arg);
        if (start.HasValue) { info.ArgumentList.Add("-ss"); info.ArgumentList.Add(start.Value.ToString("R", CultureInfo.InvariantCulture)); }
        if (duration.HasValue) { info.ArgumentList.Add("-t"); info.ArgumentList.Add(duration.Value.ToString("R", CultureInfo.InvariantCulture)); }
        foreach (string arg in new[] { "-f", "f32le", "-ac", "1", "-ar", "5512", "pipe:1" }) info.ArgumentList.Add(arg);
        using var process = Process.Start(info)!;
        using var memory = new MemoryStream();
        var warningTask = process.StandardError.ReadToEndAsync();
        await process.StandardOutput.BaseStream.CopyToAsync(memory);
        await process.WaitForExitAsync();
        string warning = await warningTask;
        if (process.ExitCode != 0) throw new IOException($"FFmpeg failed for {path}: {warning}");
        byte[] bytes = memory.ToArray();
        float[] samples = new float[bytes.Length / 4];
        Buffer.BlockCopy(bytes, 0, samples, 0, bytes.Length);
        return (samples, sw.Elapsed.TotalSeconds, warning.Length > 2048 ? warning[..2048] : warning);
    }

    static AVFingerprintConfiguration Config(bool query)
    {
        var config = new DefaultAVFingerprintConfiguration();
        if (query) config.Audio.Stride = new IncrementalRandomStride(256, 512, Seed);
        return config;
    }

    static async Task<AVHashes> Fingerprint(float[] samples, string id, bool query)
    {
        // Preserve upstream's default audio-service normalization while using
        // one shared FFmpeg decoder/resampler instead of platform WAV services.
        new AudioSamplesNormalizer().NormalizeInPlace(samples);
        var audio = new AudioSamples(samples, id, SampleRate, Epoch);
        return await FingerprintCommandBuilder.Instance.BuildFingerprintCommand()
            .From(audio).WithFingerprintConfig(Config(query)).Hash();
    }

    static object Configuration(int votes) => new
    {
        package = "SoundFingerprinting", package_version = PackageVersion,
        upstream_revision = UpstreamRevision, decoder = "FFmpeg mono float32",
        sample_rate = SampleRate, fft = 2048, fft_hop = 64,
        log_frequency_bins = 32, image_length_frames = 128, top_haar_wavelets = 200,
        minhash_permutations = 100, lsh_tables = 25, minhash_per_table = 4,
        raw_hash_payload_bytes_per_fingerprint = 100, reference_stride_samples = 512,
        query_stride_min_samples = 256, query_stride_max_exclusive_samples = 512,
        query_stride_seed = Seed, threshold_votes = votes, max_tracks_to_return = 25,
        nominal_fingerprint_length_s = 8192.0 / SampleRate,
        minimum_physical_samples_per_fingerprint = MinimumSamples,
        minimum_physical_duration_s = (double)MinimumSamples / SampleRate,
        spectral_profile = true, spectral_bridging = "NoBridgingStrategy (native default)",
        playback_speed_compensation = 0, native_audio_normalizer = true,
        external_score = "Native ResultEntry.Confidence; thresholds selected using calibration unknowns only"
        ,confidence_canonicalization_decimal_places = 12
    };

    static async Task WarmTiming(InMemoryModelService service, string inputs, string output, object setupStats)
    {
        using var metadata = JsonDocument.Parse(await File.ReadAllTextAsync(inputs));
        var clips = metadata.RootElement.GetProperty("soundfingerprinting_inputs").EnumerateArray().ToArray();
        var rows = new List<object>();
        var process = Process.GetCurrentProcess();
        foreach (var clip in clips)
        {
            byte[] raw = await File.ReadAllBytesAsync(Text(clip, "pcm_path"));
            float[] source = new float[raw.Length / 4];
            Buffer.BlockCopy(raw, 0, source, 0, raw.Length);
            var encoder = new List<double>(); var matching = new List<double>();
            var pipeline = new List<double>(); var cpu = new List<double>();
            double firstActualShapeSeconds = 0;
            int count = 0;
            for (int repeat = -2; repeat < 5; repeat++)
            {
                float[] samples = (float[])source.Clone(); //Copy/rate conversion excluded.
                TimeSpan cpuBefore = process.TotalProcessorTime;
                var clock = Stopwatch.StartNew();
                AVHashes hashes = await Fingerprint(samples, Text(clip, "query_id"), true);
                double encoded = clock.Elapsed.TotalSeconds;
                var config = new DefaultAVQueryConfiguration();
                var result = await QueryCommandBuilder.Instance.BuildQueryCommand().From(hashes)
                    .WithQueryConfig(config).UsingServices(service).Query();
                double total = clock.Elapsed.TotalSeconds;
                double cpuSeconds = (process.TotalProcessorTime - cpuBefore).TotalSeconds;
                count = hashes.Audio?.Count ?? 0;
                if (repeat == -2) firstActualShapeSeconds = total;
                if (repeat >= 0) { encoder.Add(encoded); matching.Add(total - encoded); pipeline.Add(total); cpu.Add(cpuSeconds); }
                GC.KeepAlive(result);
            }
            rows.Add(new { query_id = Text(clip, "query_id"), source_id = Text(clip, "source_id"),
                encoder_s = encoder, matching_s = matching, pipeline_s = pipeline,
                process_cpu_s = cpu, first_actual_shape_pipeline_s = firstActualShapeSeconds,
                fingerprint_count = count, sample_rate = SampleRate, sample_count = source.Length });
        }
        var outData = new { method = "SoundFingerprinting native votes4 warmed CPU", config = Configuration(4),
            common_input_sha256 = metadata.RootElement.GetProperty("common_input_sha256").GetString(),
            queries = rows, repeats = 5, warmups = 2, index_setup = setupStats,
            runtime_version = Environment.Version.ToString(), processor_count = Environment.ProcessorCount,
            included = "Native normalization/FFT/Haar/MinHash plus native default-votes4 matching",
            excluded = "Rate conversion, file I/O, input copies, index/model setup",
            note = "Per-query pipeline wall time measured directly, not sum of separate medians; CPU totals are current process only" };
        await File.WriteAllTextAsync(output, JsonSerializer.Serialize(outData, JsonOptions));
        Console.WriteLine($"Saved common50 warm timing {output}");
    }

    static async Task Main(string[] args)
    {
        if (args.Length == 0 || args.Contains("--help"))
        {
            Console.WriteLine("SoundFingerprinting 15.14.1 public benchmark bridge (native library unchanged).");
            Console.WriteLine("Required: --protocol FILE --cache DIR --native-output-dir DIR --calibrated-output-dir DIR");
            Console.WriteLine("Optional: --pcm-manifest FILE --durations 5[,2,3,10] --votes 4[,1] --query-limit N");
            Console.WriteLine("Warm profile: --warm-inputs FILE --warm-output FILE (after gallery setup; no benchmark rerun)");
            Console.WriteLine("Native full support is 1.846 s; no external query context/padding is added.");
            return;
        }
        if (args.Length % 2 != 0) throw new ArgumentException("Every option requires a value; use --help for usage");
        var options = Options(args);
        string protocolPath = Path.GetFullPath(options["--protocol"]);
        string cache = Path.GetFullPath(options["--cache"]);
        string nativeDirectory = Path.GetFullPath(options["--native-output-dir"]);
        string candidateDirectory = Path.GetFullPath(options["--calibrated-output-dir"]);
        double[] durations = options.GetValueOrDefault("--durations", "5").Split(',').Select(x => double.Parse(x, CultureInfo.InvariantCulture)).ToArray();
        int limit = int.Parse(options.GetValueOrDefault("--query-limit", int.MaxValue.ToString()));
        int[] requestedVotes = options.GetValueOrDefault("--votes", "4,1").Split(',').Select(int.Parse).ToArray();
        if (requestedVotes.Any(v => v != 4 && v != 1)) throw new ArgumentException("Supported vote variants are 4 and 1");
        Directory.CreateDirectory(cache); Directory.CreateDirectory(nativeDirectory); Directory.CreateDirectory(candidateDirectory);
        using var document = JsonDocument.Parse(await File.ReadAllTextAsync(protocolPath));
        using var pcmDocument = options.TryGetValue("--pcm-manifest", out var pcmManifest) ?
            JsonDocument.Parse(await File.ReadAllTextAsync(pcmManifest)) : null;
        var root = document.RootElement;
        string protocolHash = Convert.ToHexString(SHA256.HashData(await File.ReadAllBytesAsync(protocolPath))).ToLowerInvariant();
        if (pcmDocument != null && pcmDocument.RootElement.GetProperty("protocol_sha256").GetString() != protocolHash)
            throw new InvalidDataException("Prepared PCM manifest belongs to a different frozen protocol");
        var references = root.GetProperty("references").EnumerateArray().Where(r => new[] { "calibration_known", "test_known" }.Contains(Text(r, "role"))).ToArray();
        var queries = root.GetProperty("queries").EnumerateArray().Where(q => q.GetProperty("evaluate").GetBoolean()).Take(limit).ToArray();
        var galleryIds = references.Select(r => Text(r, "reference_id")).ToHashSet();
        foreach (var q in queries)
            if (Text(q, "role").EndsWith("unknown") && galleryIds.Contains(Text(q, "reference_id")))
                throw new InvalidDataException("Unknown source present in gallery");

        long before = GC.GetTotalMemory(true);
        var service = new InMemoryModelService();
        var buildClock = Stopwatch.StartNew();
        long fingerprints = 0, payload = 0, featureCacheBytes = 0;
        double referenceSeconds = 0, referenceExtractionSeconds = 0, referenceDecodeSeconds = 0;
        int cacheHits = 0;
        var warnings = new List<object>();
        for (int i = 0; i < references.Length; i++)
        {
            var reference = references[i];
            string id = Text(reference, "reference_id"), path = Text(reference, "path");
            string key = Hash($"{PackageVersion}|{path}|{new FileInfo(path).Length}|normalized5512_default");
            string file = Path.Combine(cache, $"reference_{key}.pb");
            AVHashes hashes;
            if (File.Exists(file))
            {
                using var stream = File.OpenRead(file);
                hashes = new AVHashes(Serializer.Deserialize<Hashes>(stream), null);
                cacheHits++;
            }
            else
            {
                JsonElement? prepared = pcmDocument?.RootElement.GetProperty("references").GetProperty(id);
                var pcm = await Decode(path, prepared: prepared);
                referenceDecodeSeconds += pcm.DecodeSeconds;
                if (pcm.Warning.Length > 0) warnings.Add(new { reference_id = id, warning = pcm.Warning });
                var extractionClock = Stopwatch.StartNew();
                hashes = await Fingerprint(pcm.Samples, id, false);
                referenceExtractionSeconds += extractionClock.Elapsed.TotalSeconds;
                using var stream = File.Create(file);
                Serializer.Serialize(stream, hashes.Audio);
            }
            service.Insert(new TrackInfo(id, reference.TryGetProperty("track_title", out var title) ? title.GetString()! : id,
                reference.TryGetProperty("artist_name", out var artist) ? artist.GetString()! : ""), hashes);
            fingerprints += hashes.Audio!.Count;
            payload += hashes.Audio.Sum(f => f.HashBins.LongLength * sizeof(int) + f.OriginalPoint.LongLength);
            referenceSeconds += hashes.Audio.DurationInSeconds;
            featureCacheBytes += new FileInfo(file).Length;
            if ((i + 1) % 25 == 0) Console.WriteLine($"SoundFingerprinting indexed {i + 1}/{references.Length}");
        }
        double indexBuildSeconds = buildClock.Elapsed.TotalSeconds;
        long after = GC.GetTotalMemory(true);
        string snapshot = Path.Combine(cache, "snapshot_" + Hash(string.Join('|', references.Select(r => Text(r, "reference_id")))));
        service.Snapshot(snapshot);
        long snapshotBytes = Directory.EnumerateFiles(snapshot, "*", SearchOption.AllDirectories).Sum(f => new FileInfo(f).Length);
        var stats = new { reference_count = references.Length, reference_fingerprint_count = fingerprints,
            reference_audio_duration_s = referenceSeconds, reference_feature_hash_payload_bytes = payload,
            reference_protobuf_cache_bytes = featureCacheBytes, index_snapshot_bytes = snapshotBytes,
            managed_index_memory_delta_bytes = Math.Max(0, after - before),
            index_build_wall_s = indexBuildSeconds, reference_extraction_work_s = referenceExtractionSeconds,
            reference_decode_s = referenceDecodeSeconds, reference_cache_hits = cacheHits,
            model_info = service.Info.Select(info => new { info.Id, info.TracksCount, info.SubFingerprintsCount, info.HashCountsInTables }),
            decoder_warnings = warnings };

        if (options.TryGetValue("--warm-inputs", out var warmInputs))
        {
            await WarmTiming(service, warmInputs, options["--warm-output"], stats);
            return;
        }

        foreach (double requested in durations)
        {
            var native = new List<Dictionary<string, object?>>();
            var candidate = new List<Dictionary<string, object?>>();
            for (int i = 0; i < queries.Length; i++)
            {
                var query = queries[i];
                double originalDuration = query.GetProperty("duration_s").GetDouble();
                double duration = Math.Min(requested, originalDuration), shift = (originalDuration - duration) / 2;
                double start = query.GetProperty("start_s").GetDouble() + shift;
                double expected = query.GetProperty("expected_reference_start_s").GetDouble() + shift * query.GetProperty("expected_time_scale").GetDouble();
                string queryId = Text(query, "query_id");
                JsonElement? prepared = pcmDocument?.RootElement.GetProperty("queries").GetProperty($"{queryId}@{requested:g}s");
                var pcm = await Decode(Text(query, "path"), start, duration, prepared);
                int wanted = (int)Math.Round(duration * SampleRate);
                if (Math.Abs(pcm.Samples.Length - wanted) > 2) throw new InvalidDataException($"Decoded crop length mismatch for {queryId}");
                var extractionClock = Stopwatch.StartNew();
                AVHashes hashes = await Fingerprint(pcm.Samples, queryId, true);
                double extractionSeconds = extractionClock.Elapsed.TotalSeconds;
                int[] order = i % 2 == 0 ? requestedVotes : requestedVotes.Reverse().ToArray();
                foreach (int votes in order)
                {
                    var queryConfig = new DefaultAVQueryConfiguration();
                    queryConfig.Audio.ThresholdVotes = votes;
                    queryConfig.Audio.Stride = new IncrementalRandomStride(256, 512, Seed);
                    var queryClock = Stopwatch.StartNew();
                    var result = await QueryCommandBuilder.Instance.BuildQueryCommand().From(hashes)
                        .WithQueryConfig(queryConfig).UsingServices(service).Query();
                    double searchSeconds = queryClock.Elapsed.TotalSeconds;
                    var best = result.Audio?.BestMatch;
                    string? predicted = best?.Track.Id;
                    double? offset = best == null ? null : best.TrackMatchStartsAt - best.QueryMatchStartsAt;
                    bool correct = predicted == Text(query, "reference_id");
                    var row = new Dictionary<string, object?> {
                        ["query_id"] = queryId, ["source_id"] = Text(query, "source_id"),
                        ["source_query_id"] = query.TryGetProperty("source_query_id", out var sq) ? sq.GetString() : Text(query, "source_id"),
                        ["role"] = Text(query, "role"), ["reference_id"] = predicted,
                        ["target_reference_id"] = Text(query, "reference_id"),
                        ["score"] = Math.Round(best?.Confidence ?? 0, 12, MidpointRounding.ToEven),
                        ["raw_score"] = best?.Confidence ?? 0,
                        ["library_similarity_score"] = best?.Score ?? 0, ["offset_s"] = offset,
                        ["expected_reference_start_s"] = expected, ["duration_s"] = duration,
                        ["correct_reference"] = correct, ["localized_correct"] = correct && Math.Abs(offset!.Value - expected) <= 2,
                        ["query_fingerprint_count"] = hashes.Audio?.Count ?? 0,
                        ["unsupported_physical_duration"] = pcm.Samples.Length < MinimumSamples,
                        ["query_decode_s"] = pcm.DecodeSeconds, ["query_extraction_s"] = extractionSeconds,
                        ["query_search_s"] = searchSeconds, ["latency_s"] = extractionSeconds + searchSeconds,
                        ["query_match_starts_at_s"] = best?.QueryMatchStartsAt,
                        ["track_match_starts_at_s"] = best?.TrackMatchStartsAt,
                        ["coverage_s"] = best?.TrackCoverageWithPermittedGapsLength };
                    foreach (string field in new[] { "condition", "speaker_id", "negative_population" })
                        if (query.TryGetProperty(field, out var value)) row[field] = value.ValueKind == JsonValueKind.Null ? null : value.GetString();
                    (votes == 4 ? native : candidate).Add(row);
                }
                if ((i + 1) % 100 == 0) Console.WriteLine($"SoundFingerprinting dual {requested:g}s queried {i + 1}/{queries.Length}");
            }
            foreach (var variant in new[] { (native, 4, nativeDirectory, "SoundFingerprinting"), (candidate, 1, candidateDirectory, "SoundFingerprinting_calibrated") })
            {
                if (!requestedVotes.Contains(variant.Item2)) continue;
                var result = new { method = variant.Item4, config = Configuration(variant.Item2), stats,
                    protocol_sha256 = protocolHash, protocol = root.GetProperty("protocol"), duration_s_requested = requested,
                    pcm_preparation_metadata = pcmDocument == null ? null : new {
                        prepare_wall_s = pcmDocument.RootElement.GetProperty("prepare_wall_s").GetDouble(),
                        prepare_workers = pcmDocument.RootElement.GetProperty("prepare_workers").GetInt32(),
                        protocol_sha256 = pcmDocument.RootElement.GetProperty("protocol_sha256").GetString() },
                    query_frontend_shared_across_variants = true,
                    latency_note = "Native normalization/fingerprint/hash extraction plus matching; FFmpeg decode separately recorded",
                    predictions = variant.Item1 };
                string file = Path.Combine(variant.Item3, $"soundfingerprinting_{requested:g}s.json");
                await File.WriteAllTextAsync(file, JsonSerializer.Serialize(result, JsonOptions));
                Console.WriteLine($"Saved {file}");
            }
        }
    }
}
