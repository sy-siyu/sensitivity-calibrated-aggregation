from __future__ import annotations

from fractions import Fraction as F
import itertools
import math
from pathlib import Path
import unittest

from experiments import jmlr_statistical_baselines_v1_core as historical_ltt_core
from experiments import jmlr_verified_adaptive_v1_math as exact
from experiments.jmlr_empirical_relevance_v1_core import (
    LTT_FRACTIONS,
    RHC_STATES,
    _choice_from_partition,
    choose_ltt_fraction,
    extended_ltt_splits,
    load_rhc,
    paired_differences,
    propensity_logit_clustering,
    rhc_split,
)
from experiments.jmlr_verified_adaptive_v1_roots import RootVerifier
from sca_reproduction.relevance import stratified_grid_summary


PROJECT = Path(__file__).resolve().parents[1]


@unittest.skipUnless((PROJECT / "data/rhc.csv").is_file(),
                     "verified RHC CSV is acquired separately")
class RHCFrozenDesignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_rhc(PROJECT / "data/rhc.csv")

    def test_documented_rows_and_declared_state_preservation(self):
        self.assertEqual(self.data["rows"], 5735)
        self.assertEqual(self.data["treated"], 2184)
        self.assertEqual(self.data["survived_30d"], 3817)
        self.assertEqual(len(RHC_STATES), 8)
        self.assertEqual(sum(self.data["state_counts"]), 5735)
        self.assertTrue(all(value > 0 for value in self.data["state_counts"]))

    def test_assignment_and_outcome_cohorts_are_disjoint_and_reproducible(self):
        first = rhc_split(self.data, 0)
        replay = rhc_split(self.data, 0)
        self.assertEqual(first, replay)
        self.assertEqual(len(first["calibration_indices"]), 4014)
        self.assertEqual(len(first["outcome_indices"]), 1721)
        self.assertFalse(set(first["calibration_indices"]) & set(first["outcome_indices"]))
        self.assertEqual(len(set(first["calibration_indices"]) |
                             set(first["outcome_indices"])), 5735)


class ClusteringTests(unittest.TestCase):
    @staticmethod
    def brute_objective(N, T, order, cuts):
        propensities = [F(2 * T[i] + 1, 2 * N[i] + 2) for i in order]
        logits = [math.log(float(value / (1 - value))) for value in propensities]
        total = 0.0
        for left, right in zip(cuts, cuts[1:]):
            weights = [N[order[j]] for j in range(left, right)]
            mass = sum(weights)
            if not mass:
                continue
            mean = sum(weight * logits[j] for weight, j in
                       zip(weights, range(left, right))) / mass
            total += sum(weight * (logits[j] - mean) ** 2 for weight, j in
                         zip(weights, range(left, right)))
        return total

    def test_dp_is_global_and_keeps_zero_weight_states(self):
        N = (10, 0, 5, 0, 8)
        T = (2, 0, 4, 0, 6)
        order = (0, 1, 2, 3, 4)
        result = propensity_logit_clustering(N, T, order, 3)
        candidates = []
        for internal in itertools.combinations(range(1, len(N)), 2):
            cuts = (0,) + internal + (len(N),)
            candidates.append((self.brute_objective(N, T, order, cuts), cuts))
        optimum = min(candidates)
        self.assertAlmostEqual(result["objective"], optimum[0], places=12)
        self.assertEqual(result["ordered_cuts"], optimum[1])
        self.assertEqual(sorted(i for cell in result["partition"] for i in cell),
                         list(range(len(N))))
        self.assertEqual(len(result["partition"]), 3)

    def test_zero_objective_tie_uses_earliest_cuts(self):
        result = propensity_logit_clustering((0, 0, 0, 0), (0, 0, 0, 0),
                                             (0, 1, 2, 3), 2)
        self.assertEqual(result["ordered_cuts"], (0, 1, 4))

    def test_certification_uses_coupled_cell_records_not_smoothed_scores(self):
        N, T = (100, 100, 100, 100), (20, 30, 50, 70)
        region = exact.confidence_region(N, T, F(1, 40), verifier=RootVerifier())
        clustering = propensity_logit_clustering(N, T, (0, 1, 2, 3), 2)
        records = {}
        choice = _choice_from_partition(region, records, clustering["partition"], 2)
        expected = max(records[cell]["coupled"] for cell in choice["partition"])
        self.assertEqual(choice["certificate"], expected)
        self.assertTrue(all(records[cell]["status"] == "full"
                            for cell in choice["partition"]))


class LTTExtensionTests(unittest.TestCase):
    def test_extended_coupling_preserves_historical_splits_and_is_nested(self):
        N = (100, 80, 60, 40)
        T = (20, 30, 25, 35)
        ids = (3, 1, 2, 0)
        historical = historical_ltt_core.nested_splits(N, T, 2, 1, 7,
                                                       immutable_ids=ids)
        extended = extended_ltt_splits(N, T, 2, 1, 7, ids)
        self.assertEqual(tuple(extended), LTT_FRACTIONS)
        for label in ("1/5", "1/2"):
            for key in ("pilot_N", "pilot_T", "heldout_N", "heldout_T"):
                self.assertEqual(tuple(extended[label][key]), tuple(historical[label][key]))
        sizes = [extended[label]["pilot_size"] for label in LTT_FRACTIONS]
        self.assertEqual(sizes, [28, 56, 93, 140, 186])
        for left, right in zip(LTT_FRACTIONS, LTT_FRACTIONS[1:]):
            self.assertTrue(all(a <= b for a, b in
                                zip(extended[left]["pilot_N"], extended[right]["pilot_N"])))

    def test_global_selection_and_tie_rule(self):
        means = {label: F(index, 100) for index, label in enumerate(LTT_FRACTIONS)}
        self.assertEqual(choose_ltt_fraction(means), "2/3")
        tied = dict(means, **{"1/10": F(1, 2), "1/3": F(1, 2), "2/3": F(1, 2)})
        self.assertEqual(choose_ltt_fraction(tied), "1/10")


class ReportingContractTests(unittest.TestCase):
    def test_paired_reporting_rejects_population_mismatch(self):
        left = [{"id": 0, "width": 0.4}, {"id": 1, "width": 0.6}]
        right = [{"id": 0, "width": 0.5}, {"id": 1, "width": 0.2}]
        differences = paired_differences(left, right, ("id",), "width")
        self.assertEqual(tuple(key for key, _ in differences), ((0,), (1,)))
        self.assertAlmostEqual(differences[0][1], -0.1)
        self.assertAlmostEqual(differences[1][1], 0.4)
        with self.assertRaises(ValueError):
            paired_differences(left, right[:1], ("id",), "width")

    def test_fixed_grid_summary_weights_conditions_not_pooled_rows(self):
        rows = [
            {"condition": "a", "value": 0.0},
            {"condition": "a", "value": 2.0},
            {"condition": "b", "value": 10.0},
            {"condition": "b", "value": 10.0},
        ]
        result = stratified_grid_summary(rows, "value")
        self.assertEqual(result["conditions"], 2)
        self.assertEqual(result["replications_per_condition"], [2])
        self.assertAlmostEqual(result["mean"], 5.5)
        self.assertGreater(result["se"], 0.0)


if __name__ == "__main__":
    unittest.main()
