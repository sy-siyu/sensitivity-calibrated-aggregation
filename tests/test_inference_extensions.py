from __future__ import annotations

from fractions import Fraction as F
import itertools
from unittest.mock import patch
import unittest

from experiments import jmlr_outcome_inference_v1_lp as historical_lp
from experiments import jmlr_verified_adaptive_v1_math as exact
from experiments.jmlr_inference_extensions_v1 import (
    affine_interval,
    analyze_release,
    bounded_mean_family,
    build_bounded_moment_region,
    choose_calibration_general,
    confidence_allocation,
    corrected_cell_certificates,
    drift_factor,
    drifted_assignment_region,
    export_release,
    hoeffding_mean_interval,
    infer_shared,
    partition_distortion,
    psi_bounds,
    release_bytes,
    sharp_finite_support_density_mean,
    validate_release,
)
from experiments.jmlr_outcome_inference_v1_core import (
    ALPHA_CAL,
    ALPHA_RAW,
    ALPHA_Z,
    choose_calibration,
    infer_outcome,
    population_interval,
    project_assignment,
)
from experiments.jmlr_verified_adaptive_v1_roots import RootPolicy, RootVerifier
from experiments.jmlr_verified_pipeline_v1_arithmetic import cell_certificate


def calibration_fixture(partition=((0, 1),), *, with_alpha=False):
    region = exact.region_from_bounds(
        (F(7, 20), F(9, 20)), (F(11, 20), F(13, 20)),
        (F(1, 6), F(2, 3)), (F(1, 3), F(5, 6)))
    if with_alpha:
        region["alpha"] = F(1, 40)
        region["root_policy"] = RootPolicy().__dict__
    selected = tuple(cell_certificate(region, cell) for cell in partition)
    return {"region": region, "partition": partition, "selected_cells": selected,
            "certificate": max(record["coupled"] for record in selected), "cap": F(10)}


def brute_density_interval(values, probabilities, budget):
    """Independent vertex enumeration for the density-box equality polytope."""
    lower, upper = F(1, budget), F(budget)
    candidates = []
    k = len(values)
    for free in range(k):
        fixed = [i for i in range(k) if i != free]
        for endpoints in itertools.product((lower, upper), repeat=len(fixed)):
            q = [None] * k
            for i, value in zip(fixed, endpoints):
                q[i] = value
            q[free] = (1 - sum(probabilities[i] * q[i] for i in fixed)) / probabilities[free]
            if lower <= q[free] <= upper:
                candidates.append(sum(probabilities[i] * q[i] * values[i] for i in range(k)))
    return min(candidates), max(candidates)


class GeneralConfidenceTests(unittest.TestCase):
    def test_default_and_general_allocations(self):
        allocation = confidence_allocation()
        self.assertEqual((allocation.alpha_cal, allocation.alpha_s, allocation.alpha_z),
                         (ALPHA_CAL, ALPHA_RAW, ALPHA_Z))
        general = confidence_allocation(F(1, 10), alpha_cal=F(1, 25),
                                        alpha_s=F(3, 100), alpha_z=F(1, 50))
        self.assertLessEqual(general.alpha_cal + general.alpha_s + general.alpha_z,
                             general.alpha)
        with self.assertRaises(ValueError):
            confidence_allocation(F(1, 20), alpha_cal=F(1, 40),
                                  alpha_s=F(1, 40), alpha_z=F(1, 40))
        with self.assertRaises(ValueError):
            confidence_allocation(F(1, 20), alpha_cal=F(1, 40))

    def test_alpha_point_zero_five_recovers_frozen_selection_and_binary_procedure(self):
        N, T = (36, 44), (11, 31)
        trees, ids = ((0, 1), (1, 0)), (10, 20)
        old_cal = choose_calibration(N, T, trees, ids)
        new_cal = choose_calibration_general(N, T, trees, ids)
        for key in ("partition", "certificate", "count", "cap", "order"):
            self.assertEqual(new_cal[key], old_cal[key])
        for key in ("mass_lower", "mass_upper", "propensity_lower", "propensity_upper",
                    "alpha", "gamma"):
            self.assertEqual(new_cal["region"][key], old_cal["region"][key])

        outcome_N, outcome_T, successes = (60, 60), (22, 38), (9, 27)
        old = infer_outcome(old_cal, outcome_N, outcome_T, successes)
        new = infer_shared(new_cal, outcome_T, successes)
        self.assertEqual(new["raw_outcome_bounds"], old["raw_outcome_bounds"])
        self.assertEqual(new["cell_outcome_bounds"], old["z_outcome_bounds"])
        for method in ("raw_stratified", "z_global", "z_cellwise", "raw_pooled"):
            self.assertEqual(new["methods_normalized"][method]["interval"],
                             old["methods"][method]["interval"])

    def test_identity_equality_requires_matching_allocations(self):
        calibration = calibration_fixture(((0,), (1,)))
        matched = infer_shared(calibration, (30, 30), (9, 21), branch="binary")
        self.assertTrue(matched["identity_matching_allocation_equality"])
        self.assertEqual(matched["methods_normalized"]["raw_stratified"]["interval"],
                         matched["methods_normalized"]["z_global"]["interval"])
        unequal = infer_shared(
            calibration, (30, 30), (9, 21), branch="binary", alpha=F(1, 10),
            alpha_cal=F(1, 25), alpha_s=F(3, 100), alpha_z=F(1, 50))
        self.assertFalse(unequal["identity_matching_allocation_equality"])
        self.assertNotEqual(unequal["raw_outcome_bounds"], unequal["cell_outcome_bounds"])


class BoundedOutcomeTests(unittest.TestCase):
    def test_hoeffding_empty_boundary_and_affine_rescaling(self):
        empty = hoeffding_mean_interval(F(0), 0, F(1, 20))
        self.assertEqual(empty["interval"], (F(0), F(1)))
        self.assertEqual(empty["status"], "structural")
        low = hoeffding_mean_interval(F(0), 100, F(1, 20))
        high = hoeffding_mean_interval(F(100), 100, F(1, 20))
        self.assertEqual(low["interval"][0], 0)
        self.assertEqual(high["interval"][1], 1)
        self.assertEqual(low["interval"][1], 1 - high["interval"][0])
        self.assertEqual(affine_interval((F(1, 4), F(3, 4)), (F(-2), F(6))),
                         (F(0), F(4)))

        calibration = calibration_fixture(((0,), (1,)))
        empty_group = infer_shared(calibration, (0, 20), (F(0), F(10)),
                                   branch="bounded")
        self.assertEqual(empty_group["raw_outcome_bounds"][0], (F(0), F(1)))
        normalized = infer_shared(calibration, (20, 20), (F(5), F(15)),
                                  branch="bounded")
        rescaled = infer_shared(calibration, (20, 20), (F(0), F(80)),
                                branch="bounded", outcome_bounds=(F(-2), F(6)))
        for method in normalized["methods"]:
            self.assertEqual(rescaled["methods"][method]["interval"],
                             affine_interval(normalized["methods_normalized"][method]["interval"],
                                             (F(-2), F(6))))
        self.assertTrue(rescaled["identity_matching_allocation_equality"])

    def test_finite_support_sharp_references_and_mean_outer_bound(self):
        fixtures = (
            ((F(0), F(1)), (F(3, 5), F(2, 5))),
            ((F(0), F(1, 2), F(1)), (F(1, 5), F(1, 2), F(3, 10))),
            ((F(1, 4), F(3, 4)), (F(2, 3), F(1, 3))),
        )
        for values, probabilities in fixtures:
            reference = sharp_finite_support_density_mean(values, probabilities, F(2))
            self.assertEqual(reference["interval"],
                             brute_density_interval(values, probabilities, 2))
            mean = sum(value * probability for value, probability in zip(values, probabilities))
            outer = psi_bounds(mean, F(2))
            self.assertLessEqual(outer[0], reference["interval"][0])
            self.assertGreaterEqual(outer[1], reference["interval"][1])
        binary_mean = F(2, 5)
        self.assertEqual(psi_bounds(binary_mean, 2),
                         sharp_finite_support_density_mean(
                             (F(0), F(1)), (1 - binary_mean, binary_mean), 2)["interval"])

    def test_nonbinary_mean_enclosure_can_be_strictly_conservative(self):
        sharp = sharp_finite_support_density_mean((F(1, 2),), (F(1),), F(2))["interval"]
        outer = psi_bounds(F(1, 2), F(2))
        self.assertEqual(sharp, (F(1, 2), F(1, 2)))
        self.assertEqual(outer, (F(1, 4), F(3, 4)))

    def test_existing_lp_constraints_are_valid_for_normalized_moments(self):
        values = (F(0), F(1, 2), F(1))
        probabilities = (F(1, 5), F(1, 2), F(3, 10))
        observed_mean = sum(value * probability for value, probability in zip(values, probabilities))
        sharp = sharp_finite_support_density_mean(values, probabilities, F(2))
        p, e = F(1), F(2, 5)
        region = build_bounded_moment_region(
            ((p, p),), ((e, e),), ((observed_mean, observed_mean),),
            ((0,),), ((observed_mean, observed_mean),), F(2))
        for missing_mean in sharp["interval"]:
            witness = (p, p * e, p * e * observed_mean, p * (1 - e) * missing_mean)
            for row, rhs in zip(region.rows, region.rhs):
                self.assertLessEqual(sum(coefficient * witness[index]
                                         for index, coefficient in row), rhs)
        solved = historical_lp.solve_enclosure(region)
        target = (e * observed_mean + (1 - e) * sharp["interval"][0],
                  e * observed_mean + (1 - e) * sharp["interval"][1])
        self.assertLessEqual(solved["interval"][0], target[0])
        self.assertGreaterEqual(solved["interval"][1], target[1])


class MassDriftAndReleaseTests(unittest.TestCase):
    def test_mass_box_projection_and_cellwise_distortion_correction(self):
        calibration = calibration_fixture()
        region, partition = calibration["region"], calibration["partition"]
        zeta = F(1, 10)
        target = drifted_assignment_region(region, zeta)
        p, pprime = (F(2, 5), F(3, 5)), (F(11, 25), F(14, 25))
        e = (F(1, 5), F(4, 5))
        for i in range(2):
            self.assertLessEqual(target["mass_lower"][i], pprime[i])
            self.assertLessEqual(pprime[i], target["mass_upper"][i])
            self.assertLessEqual(1 - zeta, pprime[i] / p[i])
            self.assertLessEqual(pprime[i] / p[i], 1 + zeta)
        pbox, ebox, _ = project_assignment(target, partition)
        target_propensity = sum(pprime[i] * e[i] for i in range(2))
        self.assertEqual(pbox, ((F(1), F(1)),))
        self.assertLessEqual(ebox[0][0], target_propensity)
        self.assertLessEqual(target_propensity, ebox[0][1])
        source_d = partition_distortion(p, e, partition)
        target_d = partition_distortion(pprime, e, partition)
        self.assertLessEqual(target_d["value"], drift_factor(zeta) * source_d["value"])
        self.assertTrue(all(new <= drift_factor(zeta) * old for new, old in
                            zip(target_d["cell_values"], source_d["cell_values"])))
        corrected = corrected_cell_certificates(partition, calibration["selected_cells"], zeta)
        self.assertEqual(corrected[0], drift_factor(zeta) * calibration["certificate"])

        result = infer_shared(calibration, (1100, 5600), (330, 3920),
                              branch="binary", zeta=zeta)
        raw_target_interval = population_interval(
            pprime, e, (F(3, 10), F(7, 10)), F(2))
        for method in ("raw_stratified", "z_global", "z_cellwise", "raw_pooled"):
            interval = result["methods_normalized"][method]["interval"]
            self.assertLessEqual(interval[0], raw_target_interval[0])
            self.assertGreaterEqual(interval[1], raw_target_interval[1])
        global_interval = result["methods_normalized"]["z_global"]["interval"]
        cellwise_interval = result["methods_normalized"]["z_cellwise"]["interval"]
        self.assertLessEqual(global_interval[0], cellwise_interval[0])
        self.assertLessEqual(cellwise_interval[1], global_interval[1])

    def test_drift_corrected_release_and_aggregate_reproduction(self):
        calibration = calibration_fixture(with_alpha=True)
        artifact = export_release(
            calibration, (20, 15), (F(32), F(18)), branch="bounded",
            outcome_bounds=(F(-2), F(6)), zeta=F(1, 20), sensitivity_ceiling=F(100))
        parsed = validate_release(artifact)
        self.assertEqual(parsed["zeta"], F(1, 20))
        self.assertEqual(artifact["outcome_mean_policy"], {
            "method": "hoeffding", "outward_decimal_precision": 80})
        self.assertNotIn("size", artifact["groups"][0])
        self.assertEqual(set(artifact["groups"][0]), {
            "label", "raw_state_count", "treated_count", "treated_outcome_sum",
            "mass_bounds", "propensity_bounds", "calibration_distortion_bound",
            "target_distortion_bound"})
        result = analyze_release(artifact)
        self.assertEqual(result["branch"], "bounded")
        self.assertTrue(result["provider_trust_required"])
        self.assertEqual(result["payload_bytes"], len(release_bytes(artifact)))
        for method in result["methods"]:
            self.assertEqual(
                result["methods"][method]["interval"],
                affine_interval(result["methods_normalized"][method]["interval"],
                                (F(-2), F(6))))
        self.assertLessEqual(result["methods"]["z_global"]["interval"][0],
                             result["methods"]["z_cellwise"]["interval"][0])
        self.assertLessEqual(result["methods"]["z_cellwise"]["interval"][1],
                             result["methods"]["z_global"]["interval"][1])
        with self.assertRaisesRegex(ValueError, "exceeds"):
            export_release(calibration, (20, 15), (F(32), F(18)), branch="bounded",
                           outcome_bounds=(F(-2), F(6)), zeta=F(1, 20),
                           sensitivity_ceiling=F(2))

    def test_bounded_release_records_and_reuses_decimal_precision(self):
        calibration = calibration_fixture(with_alpha=True)
        artifact_32 = export_release(
            calibration, (20, 15), (F(32), F(18)), branch="bounded",
            outcome_bounds=(F(-2), F(6)), sensitivity_ceiling=F(100),
            decimal_precision=32)
        artifact_80 = export_release(
            calibration, (20, 15), (F(32), F(18)), branch="bounded",
            outcome_bounds=(F(-2), F(6)), sensitivity_ceiling=F(100),
            decimal_precision=80)
        result_32 = analyze_release(artifact_32)
        result_80 = analyze_release(artifact_80)
        provider_32 = infer_shared(
            calibration, (20, 15), (F(32), F(18)), branch="bounded",
            outcome_bounds=(F(-2), F(6)), sensitivity_ceiling=F(100),
            decimal_precision=32)
        provider_80 = infer_shared(
            calibration, (20, 15), (F(32), F(18)), branch="bounded",
            outcome_bounds=(F(-2), F(6)), sensitivity_ceiling=F(100),
            decimal_precision=80)
        self.assertEqual(artifact_32["outcome_mean_policy"]["outward_decimal_precision"], 32)
        self.assertTrue(all(record["outward_decimal_precision"] == 32
                            for record in result_32["outcome_mean_evidence"]))
        self.assertNotEqual(result_32["outcome_mean_bounds"],
                            result_80["outcome_mean_bounds"])
        for method in ("z_global", "z_cellwise"):
            self.assertEqual(result_32["methods"][method]["interval"],
                             provider_32["methods"][method]["interval"])
            self.assertEqual(result_80["methods"][method]["interval"],
                             provider_80["methods"][method]["interval"])
        with self.assertRaisesRegex(ValueError, "does not match"):
            analyze_release(artifact_32, decimal_precision=80)

    def test_release_rejects_undercorrection_fractional_binary_and_extra_data(self):
        merged = calibration_fixture(with_alpha=True)
        merged_artifact = export_release(
            merged, (20, 15), (9, 6), branch="binary", zeta=F(1, 20),
            sensitivity_ceiling=F(100))
        wrong = json_clone(merged_artifact)
        wrong["groups"][0]["target_distortion_bound"] = wrong["groups"][0][
            "calibration_distortion_bound"]
        with self.assertRaises(ValueError):
            validate_release(wrong)

        calibration = calibration_fixture(((0,), (1,)), with_alpha=True)
        artifact = export_release(calibration, (20, 15), (9, 6), branch="binary",
                                  zeta=F(1, 20), sensitivity_ceiling=F(3))
        wrong = json_clone(artifact)
        wrong["groups"][0]["target_distortion_bound"] = "21/20"
        with self.assertRaises(ValueError):
            validate_release(wrong)
        wrong = json_clone(artifact)
        wrong["groups"][0]["treated_outcome_sum"] = "9/2"
        with self.assertRaises(ValueError):
            validate_release(wrong)
        wrong = json_clone(artifact)
        wrong["raw_counts"] = []
        with self.assertRaises(ValueError):
            validate_release(wrong)
        wrong = json_clone(artifact)
        wrong["root_policy"]["grid_bits"] = "48"
        with self.assertRaises(ValueError):
            validate_release(wrong)

    def test_lp_numerical_failure_retains_safe_original_unit_result(self):
        calibration = calibration_fixture(((0,), (1,)))
        with patch.object(historical_lp, "linprog", side_effect=RuntimeError("fixture")):
            result = infer_shared(calibration, (20, 20), (F(0), F(80)),
                                  branch="bounded", outcome_bounds=(F(-2), F(6)))
        self.assertEqual(result["LP"]["status"], "unresolved_solver_error")
        self.assertEqual(result["LP"]["interval"], (F(0), F(1)))
        pooled = result["methods"]["raw_pooled"]["interval"]
        self.assertGreaterEqual(pooled[0], F(-2))
        self.assertLessEqual(pooled[1], F(6))
        self.assertEqual(result["methods"]["raw_pooled"]["lp_status"],
                         "unresolved_solver_error")


def json_clone(value):
    import json
    return json.loads(json.dumps(value))


if __name__ == "__main__":
    unittest.main()
