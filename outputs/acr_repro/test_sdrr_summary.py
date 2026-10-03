"""Statistical reporting guards for the real-acoustic transfer adapter."""
import unittest
from run_sdrr_frozen import summarize, source_bootstrap

class AcousticSummaryTests(unittest.TestCase):
    def row(self,source,role,correct,accepted):
        return {'source_id':source,'creator_group':source,'role':role,'correct_reference':correct,
            'localized_correct_0p1s':False,'localized_correct_2s':correct,'accepted':accepted}

    def test_unknowns_do_not_enter_identification_denominator(self):
        rows=[self.row('known','test_known',True,True),self.row('known2','test_known',False,True),
            self.row('absent','test_unknown',True,True)]
        stats=summarize(rows)
        self.assertEqual(stats['known_queries'],2)
        self.assertEqual(stats['correct_reference_count'],1)
        self.assertEqual(stats['correct_reference_rate'],.5)
        self.assertEqual(stats['accepted_correct_count'],1)
        self.assertEqual(stats['wrong_accepted_known'],1)
        self.assertEqual(stats['false_accepts'],1)
        self.assertEqual(stats['empirical_false_accept_rate'],1.)
        self.assertEqual(stats['localized_correct_0p1s_count'],0)
        self.assertEqual(stats['localized_correct_2s_count'],1)

    def test_bootstrap_resamples_sources_with_their_correlated_queries(self):
        rows=[self.row('source1','test_known',True,True) for _ in range(3)]
        rows += [self.row('source2','test_known',False,False) for _ in range(3)]
        self.assertEqual(source_bootstrap(rows,'correct_reference',iterations=1000),[0.,1.])
        self.assertIsNone(source_bootstrap([],'correct_reference'))

if __name__=='__main__':unittest.main()
