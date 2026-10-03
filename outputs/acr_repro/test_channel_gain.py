"""The channel diagnostic removes a fixed spectral gain plus arbitrary loudness."""
import unittest
import numpy as np
from acr_fp import Fingerprinter
from calibrate_sdrr_channel import fit_log_gains

class ChannelGainTests(unittest.TestCase):
    def test_gain_direction_and_global_loudness_invariance(self):
        rng=np.random.default_rng(42)
        reference=np.exp(rng.normal(size=(216,32))).astype(np.float32)
        true_log=np.linspace(-1,1,32)
        ratios=[]
        for amplitude in [.2,1.,5.]:
            phone=reference/(np.exp(true_log)[None,:]*amplitude)
            ratios.append(np.log(reference.mean(axis=0,dtype=np.float64)/phone.mean(axis=0,dtype=np.float64)))
        estimated=fit_log_gains(ratios)
        np.testing.assert_allclose(estimated,true_log,atol=1e-7)
        corrected=phone*np.exp(estimated)[None,:]
        fp=Fingerprinter()
        before,t=fp.raw_features_from_mels(reference)
        after,u=fp.raw_features_from_mels(corrected.astype(np.float32))
        np.testing.assert_allclose(before,after,atol=1e-6,rtol=2e-5)
        np.testing.assert_array_equal(t,u)

    def test_invalid_or_nonfinite_gains_fail(self):
        with self.assertRaises(ValueError):fit_log_gains(np.ones((10,31)))
        invalid=np.zeros((10,32));invalid[0,0]=np.nan
        with self.assertRaises(ValueError):fit_log_gains(invalid)

if __name__=='__main__':unittest.main()
