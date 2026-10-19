from fractions import Fraction as Q
from itertools import product
import math
import unittest

from experiments import jmlr_verified_adaptive_v1_math as exact
from experiments.jmlr_verified_adaptive_v1_roots import RootVerifier
from experiments.jmlr_certification_bottleneck_v1_core import (
    sidak_individual_error,sidak_confidence_region,certified_exp_threshold,
    one_cell_distortion,lower_bound_record,boundary_alternative,
    sidak_manuscript_bound,analytical_crossing)
from experiments.jmlr_verified_pipeline_v1_core import select_families
from sca_reproduction.certification import cluster_summary


class SidakAllocationTests(unittest.TestCase):
    def test_exact_product_allocation_is_conservative_and_tighter(self):
        for alpha_e in (Q(1,20),Q(1,40),Q(1,80)):
            for K in (2,16,64):
                answer=sidak_individual_error(alpha_e,K)
                self.assertGreater(answer["eta"],alpha_e/K)
                self.assertGreaterEqual((1-answer["eta"])**K,1-alpha_e)
                self.assertTrue(answer["exact_product_verified"])

    def test_unknown_mass_keeps_mass_bonferroni_separate(self):
        N=(50,60,70);T=(20,30,40);alpha=Q(1,20)
        region=sidak_confidence_region(N,T,alpha,verifier=RootVerifier())
        self.assertEqual(region["alpha_e"],Q(1,40));self.assertEqual(region["alpha_p"],Q(1,40))
        self.assertEqual(region["mass_gamma"],Q(1,120))
        self.assertGreater(region["propensity_gamma"],Q(1,120))
        self.assertNotIn("gamma",region)
        known=sidak_confidence_region(N,T,alpha,(Q(1,3),)*3,RootVerifier())
        self.assertEqual(known["alpha_e"],alpha);self.assertEqual(known["alpha_p"],0)
        self.assertIsNone(known["mass_gamma"])

    def test_sidak_endpoints_are_no_wider_than_bonferroni(self):
        N=(100,120);T=(40,70);alpha=Q(1,20);verifier=RootVerifier()
        sidak=sidak_confidence_region(N,T,alpha,known_masses=(Q(1,2),Q(1,2)),verifier=verifier)
        bonf=exact.confidence_region(N,T,alpha,known_masses=(Q(1,2),Q(1,2)),verifier=verifier)
        for sl,su,bl,bu in zip(sidak["propensity_lower"],sidak["propensity_upper"],
                               bonf["propensity_lower"],bonf["propensity_upper"]):
            self.assertGreaterEqual(sl,bl);self.assertLessEqual(su,bu)


class CertifiedMarginTests(unittest.TestCase):
    def test_exact_and_exponential_boundaries_are_resolved(self):
        values=(Q(1),Q(3,2),Q(149,100),Q(8,5))
        exact_zero=certified_exp_threshold(Q(3,2),Q(0),values)
        self.assertEqual(exact_zero["classification"],(True,True,True,False))
        positive=certified_exp_threshold(Q(3,2),Q(1,40),values)
        self.assertEqual(positive["classification"],(True,False,False,False))
        self.assertTrue(positive["boundary_comparisons_certified"])

    def test_inapplicable_threshold_below_one_is_flagged(self):
        answer=certified_exp_threshold(Q(1),Q(1,100),(Q(1),))
        self.assertFalse(answer["applicable"])
        self.assertEqual(answer["reason"],"threshold-below-one")

    def test_small_support_oracle_count_matches_exhaustive_realized_family(self):
        def contiguous_partitions(order):
            K=len(order)
            for cuts in product((False,True),repeat=max(K-1,0)):
                start=0;partition=[]
                for position,cut in enumerate(cuts,1):
                    if cut:
                        partition.append(tuple(sorted(order[start:position])));start=position
                partition.append(tuple(sorted(order[start:])))
                yield exact.canonical_partition(partition)

        def tree_partitions(tree,cell):
            if tree[cell] is None:
                return {(cell,)}
            left,right=tree[cell]
            answer={exact.canonical_partition((cell,))}
            for a in tree_partitions(tree,left):
                for b in tree_partitions(tree,right):
                    answer.add(exact.canonical_partition(a+b))
            return answer

        for K in range(1,7):
            order=tuple(range(K))[1::2]+tuple(range(K))[::2]
            tree_orders=(order,tuple(reversed(order)))
            trees=tuple(exact.build_tree(tree_order) for tree_order in tree_orders)
            partitions=set(contiguous_partitions(order))
            for tree in trees:
                partitions.update(tree_partitions(tree,max(tree,key=len)))
            cells={cell for partition in partitions for cell in partition}
            # Singletons are the identity; non-singletons span both sides of the
            # positive-margin thresholds in a deterministic exact-rational pattern.
            scores={cell:(Q(1) if len(cell)==1 else
                (Q(6,5),Q(3,2),Q(9,5),Q(11,5))[(sum(cell)+len(cell))%4])
                for cell in cells}
            for cap,delta in ((Q(3,2),Q(0)),(Q(2),Q(1,20)),(Q(2),Q(1,10))):
                threshold=certified_exp_threshold(cap,delta,tuple(scores.values()))
                self.assertTrue(threshold["applicable"])
                separator=threshold["separator"]
                feasible=[partition for partition in partitions
                          if all(scores[cell] <= separator for cell in partition)]
                self.assertTrue(feasible)
                selected=select_families(order,trees,scores,separator)["combined"]
                self.assertEqual(selected["cells"],min(map(len,feasible)))
                self.assertTrue(all(scores[cell] <= separator
                                    for cell in selected["partition"]))


class ReportingTests(unittest.TestCase):
    def test_equal_condition_cluster_summary_preserves_replication_pairing(self):
        rows=[{"replication":rep,"metric":value}
              for rep,values in ((0,(0,2)),(1,(10,12)),(2,(20,22)))
              for value in values]
        answer=cluster_summary(rows,"metric",condition_count=2,replications=3)
        self.assertEqual(answer["n"],3)
        self.assertEqual(answer["mean"],11)
        self.assertAlmostEqual(answer["se"],10/math.sqrt(3))


class SparseMixtureTests(unittest.TestCase):
    def test_event_projection_lower_bound_on_declared_rectangle(self):
        alpha,gamma=Q(1,20),Q(1,10)
        target=(1-gamma-alpha)**2/(gamma*(1-gamma))
        for p in (Q(9,10),Q(19,20),Q(99,100)):
            for q in (Q(0),Q(1,40),alpha):
                self.assertGreaterEqual((p-q)**2/(p*(1-p)),target)

    def test_exact_small_case_enumeration_matches_second_moment(self):
        K,n=3,2;a,b=Q(1,2),Q(1,3)
        alphabet=tuple(product(range(K),range(2)))
        def one_prob(model,x):
            s,y=x;e=b if model==s else a
            return Q(1,K)*(e if y else 1-e)
        p0=lambda x:Q(1,K)*(a if x[1] else 1-a)
        chi=Q(0)
        for sequence in product(alphabet,repeat=n):
            p=math.prod(p0(x) for x in sequence)
            q=sum(math.prod(one_prob(j,x) for x in sequence) for j in range(K))/K
            chi+=(q-p)**2/p
        kappa=(b-a)**2/(a*(1-a))
        self.assertEqual(chi,((1+kappa/K)**n-1)/K)

    def test_reference_values_and_strict_boundary(self):
        first=lower_bound_record(64,Q(1,2),Q(4,13),Q(3,2),Q(1,20),Q(1,10))
        second=lower_bound_record(64,Q(1,2),Q(199,500),Q(3,2),Q(1,20),Q(1,10))
        self.assertEqual(first["distortion"],Q(827,372));self.assertEqual(first["mixture_ceiling"],2705)
        self.assertEqual(first["two_point_ceiling"],1900)
        self.assertEqual(second["mixture_ceiling"],9606);self.assertEqual(second["two_point_ceiling"],7156)
        self.assertTrue(first["two_point_ceiling_certificate"]["adjacent_integer_inequalities_certified"])
        self.assertTrue(first["mixture_ceiling_certificate"]["adjacent_integer_inequalities_certified"])
        self.assertTrue(second["two_point_ceiling_certificate"]["adjacent_integer_inequalities_certified"])
        self.assertTrue(second["mixture_ceiling_certificate"]["adjacent_integer_inequalities_certified"])
        boundary=boundary_alternative(64,Q(1,2),Q(3,2),Q(1,4),10**12)
        self.assertGreater(boundary["distortion"],Q(3,2));self.assertTrue(boundary["grid_maximal"])

    def test_sidak_analytical_crossing_is_discrete(self):
        crossing=analytical_crossing(lambda n:sidak_manuscript_bound(
            n,64,Q(1,20),Q(1,64),Q(1,4),known_mass=False,beta=Q(1,20)))
        self.assertLess(crossing["previous_integer"]["b_n"],1) # finite near crossing
        self.assertGreaterEqual(crossing["previous_integer"]["b_n"],math.log(1.5))
        self.assertLess(crossing["at_threshold"]["b_n"],math.log(1.5))


if __name__=="__main__":unittest.main()
