from fractions import Fraction as F
import unittest

from experiments.jmlr_five_state_counterexample_v1 import CAP, verify


class FiveStateCounterexampleTests(unittest.TestCase):
    def test_bounded_search_and_independent_exact_enumerations(self):
        receipt = verify()
        self.assertEqual(CAP, F(19, 10))
        self.assertLessEqual(CAP, 2)
        self.assertEqual(receipt["enumeration"]["unrestricted_partitions"], 52)
        self.assertEqual(receipt["enumeration"]["contiguous_cut_partitions"], 16)
        self.assertEqual(receipt["optima"]["unrestricted_count"], 2)
        self.assertEqual(receipt["optima"]["contiguous_count"], 3)
        self.assertEqual(receipt["optima"]["unrestricted_best_two_distortion"],
                         "25244179/13307679")
        self.assertEqual(receipt["optima"]["contiguous_best_two_distortion"],
                         "12892811/6780186")
        self.assertEqual(receipt["optima"]["contiguous_best_three_distortion"],
                         "435789/341539")


if __name__ == "__main__":
    unittest.main()
