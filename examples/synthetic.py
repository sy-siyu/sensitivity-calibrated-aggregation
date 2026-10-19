"""A short full-sample sensitivity-calibrated aggregation (SCA) workflow."""
import json
import numpy as np
from sca import SCA, analyze_outcome_release


def main():
    K = 8
    p, e = np.ones(K)/K, np.linspace(.3, .7, K)
    calibration = np.random.default_rng(2026101801)
    N = calibration.multinomial(4000*K, p)
    T = calibration.binomial(N, e)
    fitted = SCA().fit(N, T, cap='3/2')
    # Freeze the grouping before drawing an independent outcome cohort.
    # Both samples use the same population masses p and propensities e,
    # hence the same P(S, A); their observed counts need not match.
    outcomes = np.random.default_rng(2026101802)
    out_N = outcomes.multinomial(256*K, p)
    out_T = outcomes.binomial(out_N, e)
    successes = outcomes.binomial(out_T, .5)
    artifact = fitted.release_outcomes(out_T, successes, branch="binary")
    analyst = analyze_outcome_release(artifact)
    print(json.dumps({'procedure':'sensitivity-calibrated aggregation (SCA)',
        'objective':fitted.objective,'groups':fitted.groups,
        'distortion_certificate':round(float(fitted.certificate), 6),
        'optimization_status':fitted.status,
        # Rounded display only; the solver and release keep exact fractions.
        'global_interval':[round(float(x), 6) for x in analyst['methods']['z_global']['interval']],
        'cellwise_interval':[round(float(x), 6) for x in analyst['methods']['z_cellwise']['interval']]},indent=2))


if __name__=='__main__':main()
