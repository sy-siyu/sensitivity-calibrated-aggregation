"""Small supplied-mass-drift release with corrected admission checks."""
import json

from sca import SCA, analyze_outcome_release


fit = SCA().fit((80, 80), (24, 56), cap='10')
release = fit.release_outcomes(
    outcome_treated=(20, 20),
    treated_outcome_sums=(8, 12),
    branch='binary',
    mass_drift='1/20',
    reporting_capacity=1,
    sensitivity_ceiling='100',
)
analysis = analyze_outcome_release(release)

rejected = None
try:
    fit.release_outcomes(
        (20, 20), (8, 12), branch='binary', mass_drift='1/20',
        reporting_capacity=1, sensitivity_ceiling='2')
except ValueError as error:
    rejected = str(error)

print(json.dumps({
    'mass_drift': release['sensitivity']['zeta'],
    'corrected_certificate': str(analysis['corrected_certificate']),
    'transported_budget': str(analysis['transported_budget']),
    'admission_verified': analysis['admission_verified'],
    'tight_ceiling_rejection': rejected,
}, indent=2))
