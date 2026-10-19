from fractions import Fraction as F
import unittest

from sca import SCA, analyze_outcome_release


class PublicInferenceAPITests(unittest.TestCase):
    def setUp(self):
        self.N = (80, 80)
        self.T = (24, 56)

    def test_existing_binary_release_call_remains_available(self):
        result = SCA().fit(self.N, self.T, cap='3/2')
        artifact = result.release((60, 60), (20, 40), (8, 30))
        self.assertEqual(artifact['version'], 'jmlr_release_capacity_v1')

    def test_bounded_release_reconstructs_with_recorded_precision(self):
        result = SCA().fit(self.N, self.T, cap='10')
        artifact = result.release_outcomes(
            (30, 40), (F(25), F(65)), branch='bounded', outcome_bounds=('-2', '3'),
            reporting_capacity=1, sensitivity_ceiling='100', decimal_precision=40)
        self.assertEqual(artifact['selection']['reporting_capacity'], 1)
        self.assertEqual(artifact['outcome_mean_policy']['outward_decimal_precision'], 40)
        reconstructed = analyze_outcome_release(artifact)
        provider = result.infer_outcomes(
            (30, 40), (F(25), F(65)), branch='bounded', outcome_bounds=('-2', '3'),
            sensitivity_ceiling='100', decimal_precision=40)
        for method in ('z_global', 'z_cellwise'):
            self.assertEqual(reconstructed['methods'][method]['interval'],
                             provider['methods'][method]['interval'])
        self.assertTrue(reconstructed['admission_verified'])
        with self.assertRaisesRegex(ValueError, 'does not match'):
            analyze_outcome_release(artifact, decimal_precision=80)

    def test_capacity_and_drift_corrected_ceiling_are_enforced(self):
        identity = SCA().fit(self.N, self.T, cap='1')
        with self.assertRaisesRegex(ValueError, 'reporting capacity'):
            identity.release_outcomes((20, 20), (8, 12), reporting_capacity=1)

        merged = SCA().fit(self.N, self.T, cap='10')
        with self.assertRaisesRegex(ValueError, 'exceeds'):
            merged.release_outcomes((20, 20), (8, 12), reporting_capacity=1,
                                    mass_drift='1/20', sensitivity_ceiling='2')

    def test_calibration_and_overall_errors_are_distinct(self):
        result = SCA(alpha='1/40').fit(self.N, self.T, cap='10')
        inferred = result.infer_outcomes(
            (20, 20), (8, 12), overall_alpha='1/10', alpha_cal='1/40',
            alpha_raw='3/80', alpha_aggregate='3/80', sensitivity_ceiling='100')
        self.assertEqual(inferred['allocation'].alpha, F(1, 10))
        self.assertEqual(inferred['allocation'].alpha_cal, F(1, 40))
        with self.assertRaisesRegex(ValueError, 'different alpha'):
            result.infer_outcomes((20, 20), (8, 12), overall_alpha='1/10')


if __name__ == '__main__':
    unittest.main()
