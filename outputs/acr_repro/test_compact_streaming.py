"""Conformance tests for the store experiment; no benchmark-dependent choices."""
import sys
from pathlib import Path
import unittest
import tempfile
import numpy as np
import faiss
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'outputs/acr_repro'))
from acr_fp import ExactIndex, FingerprintConfig
from compact_streaming_index import CompactTemporalIndex, _ContentLookup, _TimeLookup

class CompactStoreTests(unittest.TestCase):
    def test_boundary_lookup_including_empty_content(self):
        lookup = _ContentLookup(np.array([3, 3, 7, 10]))
        np.testing.assert_array_equal(lookup[np.array([0, 2, 3, 6, 7, 9])], [0, 0, 2, 2, 3, 3])

    def test_physical_clock_bit_parity_after_silence_gaps(self):
        config = FingerprintConfig()
        frames = np.array([0, 8, 16, 80, 80000, 1000000], dtype=np.uint32)
        expected = frames.astype(np.float64) * config.hop_length / config.sample_rate + config.first_center_seconds
        actual = _TimeLookup(frames, config.hop_length, config.sample_rate, config.first_center_seconds)[np.arange(len(frames))]
        np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))

    def test_exact_neighbours_and_vote_parity(self):
        config = FingerprintConfig()
        rng = np.random.default_rng(42)
        vectors = rng.normal(size=(24, 32)).astype(np.float32)
        frames = np.tile(np.arange(0, 64, 8), 3).astype(np.uint32)
        times = frames.astype(np.float64) * config.hop_length / config.sample_rate + config.first_center_seconds
        baseline = ExactIndex(vectors, np.repeat(np.arange(3), 8), times, ['a', 'b', 'c'], factor=8)
        index = faiss.IndexFlatL2(32); index.add(vectors)
        compact = CompactTemporalIndex(index, frames, [8, 16, 24], ['a', 'b', 'c'], config)
        query = vectors[8:16].copy()
        qt = times[8:16] - 2.
        for left, right in zip(baseline.search(query), compact.search(query)):
            np.testing.assert_array_equal(left, right)
        self.assertEqual(baseline.match(query, qt), compact.match(query, qt))
        self.assertEqual(compact.metadata_bytes(), frames.nbytes + 3 * 8 + 3)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            faiss.write_index(index,str(path/'index.faiss'))
            compact.save_metadata(path/'metadata.npz')
            loaded=CompactTemporalIndex.load(path/'index.faiss',path/'metadata.npz')
            self.assertEqual(loaded.match(query,qt),baseline.match(query,qt))

    def test_invalid_metadata_rejected(self):
        index = faiss.IndexFlatL2(32); index.add(np.zeros((2, 32), dtype=np.float32))
        with self.assertRaises(ValueError):
            CompactTemporalIndex(index, [0], [2], ['a'], FingerprintConfig())
        with self.assertRaises(ValueError):
            CompactTemporalIndex(index, [0, 8], [1], ['a'], FingerprintConfig())

if __name__ == '__main__':
    unittest.main()
