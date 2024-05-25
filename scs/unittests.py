import unittest
import numpy as np
import data_preparation as dp
import snr_test_new as snrt
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

class Tester(unittest.TestCase):


    def testnoisextract(self):
        df = dp.load_dataset("../data/raw/sn_data.parquet")
        df = df.iloc[:10]
        df_injected = snrt.inject_noise(1, plot=False, recalculate=True)
        df.filter(regex="\d+")
        vec = np.vectorize(np.testing.assert_array_almost_equal, signature="(n),(n)->()")
            #self.assertAlmostEqual, signature="(n),(n)->()")
        vec(df.filter(regex="\d+").values,
            df_injected.filter(regex="\d+").values)#, places=7,

    def testinvertrfftfreqeodd(self):
        "unit test for utility function invertrfftfreq"
        x = np.arange(0, 9, 0.2)

        self.assertListEqual(x.tolist(),
            snrt.invertrfftfreq(x, 0.2).tolist(),
            f"testinvertrfftfreq failed with x % 2 == {len(x) % 2}")

    def testinvertrfftfreqeeven(self):
        "unit test for utility function invertrfftfreq"
        x = np.arange(0, 9, 0.2)

        x = x[:-1]
        self.assertListEqual(x.tolist(),
                        snrt.invertrfftfreq(x,
                                   0.2).tolist(),
                              f"testinvertrfftfreq failed with x % 2 == {len(x) % 2}")
if __name__ == "__main__":
    unittest.main()