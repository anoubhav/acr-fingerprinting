"""Integrity checks for source exclusion and rare-event reporting."""
import unittest
from protocol import assign_roles,split_manifest
from summarize_results import select_threshold,wilson,crossed_interval

class EvaluationTest(unittest.TestCase):
    def test_order_independent_source_roles(self):
        ids=[f"t{x:03d}" for x in range(100)]
        roles=assign_roles(ids)
        self.assertEqual(roles,assign_roles(list(reversed(ids))))
        self.assertEqual(sum(v=="pca_fit" for v in roles.values()),10)
        self.assertEqual(sum(v=="test_known" for v in roles.values()),55)

    def test_unknown_overlap_never_counts_as_absent(self):
        ids=[f"t{x:03d}" for x in range(100)]
        roles=assign_roles(ids)
        unknown=next(i for i in ids if roles[i]=="test_unknown")
        known=next(i for i in ids if roles[i]=="test_known")
        manifest={"summary":{},"references":[{"source_id":i,"reference_id":i} for i in ids],
            "queries":[{"source_id":unknown,"reference_id":unknown,
                "overlapping_other_annotations":[{"reference_id":known}]}]}
        result=split_manifest(manifest)
        self.assertFalse(result["queries"][0]["evaluate"])

    def test_zero_observed_fpr_has_nonzero_uncertainty(self):
        interval=wilson(0,71)
        self.assertAlmostEqual(interval[0],0)
        self.assertGreater(interval[1],.05)
        self.assertLess(interval[1],.052)
        negatives=[{"score":0.0} for _ in range(74)]
        threshold=select_threshold(negatives)
        self.assertGreater(threshold,0)
        self.assertFalse(any(x["score"]>=threshold for x in negatives))

    def test_crossed_bootstrap_point_estimate(self):
        records=[{"source_id":str(i//2),"source_query_id":str(i%2),
                  "correct":i%2==0} for i in range(12)]
        result=crossed_interval(records,"correct",replicates=500)
        self.assertEqual(result["estimate"],.5)
        self.assertEqual(result["n_source_clusters"],6)
        self.assertEqual(result["n_montage_clusters"],2)
        self.assertLessEqual(result["ci95"][0],.5)
        self.assertGreaterEqual(result["ci95"][1],.5)

if __name__=="__main__":
    unittest.main()
