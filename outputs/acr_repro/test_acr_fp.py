"""Meaningful numerical and retrieval checks. Run: python -m unittest -v test_acr_fp."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
import numpy as np
from acr_fp import FingerprintConfig, Fingerprinter, ExactIndex, standardize_rows


class FingerprintTests(unittest.TestCase):
    def setUp(self):
        self.config = FingerprintConfig()
        rng = np.random.default_rng(42)
        self.audio = (rng.standard_normal(24000) * .15).astype(np.float32)

    def test_recovered_source_formula_against_librosa(self):
        # Independent library STFT/filterbank and direct 3-D window averaging
        # are the oracle for the source notebook's pre-PCA transformations.
        import librosa
        c, fp = self.config, Fingerprinter(self.config)
        mels = librosa.feature.melspectrogram(y=np.pad(self.audio, (512, 512)),
                    sr=8000, n_fft=1024, hop_length=186, n_mels=32,
                    fmin=315, fmax=1960, center=False, power=1).T
        np.testing.assert_allclose(fp.mel_spectrogram(self.audio, 8000), mels, rtol=2e-6, atol=2e-7)
        means = np.lib.stride_tricks.sliding_window_view(mels, 32, axis=0).mean(axis=2)
        keep = (means.std(axis=1) > 1e-6) & (means.mean(axis=1) > 1e-3)
        means = means[keep]
        differences = np.diff(means, axis=1)
        z = (means - means.mean(axis=1, keepdims=True)) / means.std(axis=1, keepdims=True)
        dz = (differences - differences.mean(axis=1, keepdims=True)) / differences.std(axis=1, keepdims=True)
        expected = np.hstack([z, dz])
        actual, times = fp.raw_features(self.audio, 8000)
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=8e-6)
        self.assertAlmostEqual(times[0], .360375)
        self.assertAlmostEqual(c.support_seconds, .84875)
        self.assertAlmostEqual(c.fingerprint_rate, 8000 / 186)

    def test_gain_and_silence(self):
        fp = Fingerprinter(self.config)
        a, _ = fp.raw_features(self.audio, 8000)
        b, _ = fp.raw_features(self.audio * .4, 8000)
        np.testing.assert_allclose(a, b, rtol=2e-5, atol=3e-6)
        silence, times = fp.raw_features(np.zeros(24000), 8000)
        self.assertEqual(silence.shape, (0, 63))
        self.assertEqual(len(times), 0)
        short, _ = fp.raw_features(np.ones(100), 8000)
        self.assertEqual(len(short), 0)
        self.assertTrue(np.isfinite(standardize_rows(np.ones((2, 32)))).all())

    def test_fft_chunk_boundary_preserves_spectral_rows(self):
        import librosa
        audio = np.tile(self.audio, 20)
        fp = Fingerprinter(self.config)
        mel = fp.mel_spectrogram(audio, 8000)
        self.assertGreater(len(mel), 2048)
        oracle = librosa.feature.melspectrogram(y=np.pad(audio, (512, 512)),
                    sr=8000, n_fft=1024, hop_length=186, n_mels=32,
                    fmin=315, fmax=1960, center=False, power=1).T
        np.testing.assert_allclose(mel[2040:2060], oracle[2040:2060], rtol=2e-6, atol=2e-7)

    def test_silence_removal_preserves_clock_and_skip_phase(self):
        fp = Fingerprinter(self.config)
        audio = np.r_[self.audio[:8000], np.zeros(16000), self.audio[:8000]]
        x, t = fp.raw_features(audio, 8000)
        self.assertGreater(np.max(np.diff(t)), .5)
        index = ExactIndex.from_entries([("clip", x, t)], factor=8,
                    grid_step_sec=186/8000, grid_origin_sec=.360375)
        frames = np.rint((index.times - .360375) / (186/8000)).astype(int)
        self.assertTrue(np.all(frames % 8 == 0))
        # Source skip=7 applies to windows before filtering, so independent
        # stride-eight extraction must select the same rows and timestamps.
        x8, t8 = Fingerprinter(replace(self.config, stride_frames=8)).raw_features(audio, 8000)
        np.testing.assert_allclose(index.vectors, x8)
        np.testing.assert_allclose(index.times, t8)

    def test_pca_serialization_and_split_guard(self):
        fp = Fingerprinter(self.config)
        x, _ = fp.raw_features(self.audio, 8000)
        fp.fit([x], content_ids=np.array(["calibration"]))
        transformed = fp.transform(x)
        self.assertEqual(transformed.shape[1], 32)
        self.assertLessEqual(fp.explained_variance_ratio.sum(), 1 + 1e-12)
        fp.assert_disjoint(reference_ids=["reference"], query_ids=["query"])
        with self.assertRaises(ValueError):
            fp.assert_disjoint(reference_ids=["calibration"])
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / "model.npz"
            fp.save(p)
            reloaded = Fingerprinter.load(p)
            np.testing.assert_array_equal(transformed, reloaded.transform(x))
            self.assertEqual(reloaded.fit_content_ids, ["calibration"])

    def test_exact_distance_and_index_bytes_roundtrip(self):
        rng = np.random.default_rng(1)
        ref = rng.standard_normal((100, 32)).astype(np.float32)
        q = rng.standard_normal((7, 32)).astype(np.float32)
        index = ExactIndex.from_entries([("a", ref, np.arange(100) / 40)])
        d, ids = index.search(q, k=5)
        direct = np.sum((q[:, None] - ref[None]) ** 2, axis=2)
        expected_ids = np.argsort(direct, axis=1)[:, :5]
        np.testing.assert_array_equal(ids, expected_ids)
        np.testing.assert_allclose(d, np.take_along_axis(direct, expected_ids, axis=1), rtol=2e-6, atol=2e-5)
        self.assertEqual(index.bytes()["array_bytes"], 100 * (128 + 4 + 8))
        half = ExactIndex.from_entries([("a", ref, np.arange(100) / 40)], storage_dtype="float16")
        self.assertEqual(half.bytes()["payload_bytes"], 100 * 64)
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / "index.npz"
            index.save(p)
            reloaded = ExactIndex.load(p)
            np.testing.assert_array_equal(reloaded.search(q, 5)[1], ids)

    def test_temporal_matching_coarse_grid_and_unique_votes(self):
        # Continuous trajectory, independent non-grid query start, and 8x
        # thinning verify clock consensus rather than exact frame equality.
        def curve(t):
            return np.column_stack([np.sin(2*t), np.cos(2*t), t*.2]).astype(np.float32)
        times = .360375 + np.arange(200) * (186 / 8000)
        rng = np.random.default_rng(10)
        distractor = rng.standard_normal((200, 3)).astype(np.float32) + 3
        entries = [("true", curve(times), times), ("wrong", distractor, times)]
        qt = .360375 + np.arange(35) * (186 / 8000)
        start = 1.137
        query = curve(qt + start)
        for factor in (1, 2, 4, 6, 8):
            index = ExactIndex.from_entries(entries, factor=factor,
                        grid_step_sec=186/8000, grid_origin_sec=.360375)
            candidate = index.match(query, qt, top_k=5, tolerance_sec=.12)[0]
            self.assertEqual(candidate["content_id"], "true")
            self.assertEqual(candidate["votes"], len(qt))
            self.assertLess(abs(candidate["offset_sec"] - start), .1)
            self.assertLessEqual(candidate["max_time_residual_sec"], .12 + 1e-12)

    def test_missing_ann_neighbors_do_not_vote_for_last_reference(self):
        index = ExactIndex.from_entries([("a", np.ones((5,3),np.float32), np.arange(5)*.1)])
        class Missing:
            def search(self,q,k):
                return np.full((len(q),k),np.finfo(np.float32).max,np.float32), np.full((len(q),k),-1,np.int64)
        index._faiss = Missing()
        self.assertEqual(index.match(np.ones((3,3),np.float32),np.arange(3)*.1),[])

    def test_empirical_far_threshold_and_query_grid(self):
        from study_query_ivf import threshold_from_unknown, thin_query
        scores = np.arange(100)/100
        threshold = threshold_from_unknown(scores)
        self.assertEqual(np.count_nonzero(scores>=threshold),1)
        scores = np.arange(74)/100
        threshold = threshold_from_unknown(scores)
        self.assertEqual(np.count_nonzero(scores>=threshold),0)
        tied = np.ones(100)
        self.assertEqual(np.count_nonzero(tied>=threshold_from_unknown(tied)),0)
        with self.assertRaises(ValueError):
            threshold_from_unknown([])
        t = self.config.first_center_seconds + np.array([0,1,3,4,7,8])*(186/8000)
        x = np.arange(6*3).reshape(6,3)
        xx,tt = thin_query(x,t,4,self.config)
        np.testing.assert_array_equal(xx,x[[0,3,5]])
        np.testing.assert_array_equal(tt,t[[0,3,5]])

    def test_numeric_crop_duration_has_one_cache_key(self):
        from run_acr_protocol import query_crop
        from run_ablations import query_path
        q={"query_id":"test","path":"test.wav","start_s":0.,"duration_s":10.,
           "expected_reference_start_s":12.,"expected_time_scale":1.}
        integer=query_crop(q,5)
        floating=query_crop(q,5.)
        self.assertIsInstance(integer["duration_s"],float)
        self.assertEqual(integer,floating)
        self.assertEqual(query_path("cache",integer,self.config),query_path("cache",floating,self.config))

    def test_affine_time_estimation_without_oracle_tempo(self):
        from affine_match import match_affine
        def curve(t):
            return np.column_stack([np.sin(2*t),np.cos(2*t),t*.2]).astype(np.float32)
        rt=self.config.first_center_seconds+np.arange(500)*(186/8000)
        index=ExactIndex.from_entries([("true",curve(rt),rt)],factor=8,
                    grid_step_sec=186/8000,grid_origin_sec=self.config.first_center_seconds)
        qt=self.config.first_center_seconds+np.arange(47)*4*(186/8000)
        q=curve(1.137+1.2*qt)
        unit=index.match(q,qt)[0]
        affine=match_affine(index,q,qt)[0]
        self.assertGreater(affine["votes"],unit["votes"])
        self.assertGreaterEqual(affine["votes"],len(qt)-1)
        self.assertLess(abs(affine["time_scale"]-1.2),.035)
        self.assertLess(abs(affine["offset_sec"]-1.137),.1)


if __name__ == "__main__":
    unittest.main()
