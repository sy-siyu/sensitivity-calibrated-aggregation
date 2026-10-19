"""Small bounded-outcome release and analyst reconstruction."""
from fractions import Fraction as F
import json

from sca import SCA, analyze_outcome_release


fit = SCA().fit((80, 80), (24, 56), cap='10')
release = fit.release_outcomes(
    outcome_treated=(30, 40),
    treated_outcome_sums=(F(25), F(65)),
    branch='bounded',
    outcome_bounds=('-2', '3'),
    reporting_capacity=1,
    sensitivity_ceiling='100',
    decimal_precision=40,
)
analysis = analyze_outcome_release(release)
print(json.dumps({
    'groups': len(release['groups']),
    'reporting_capacity': release['selection']['reporting_capacity'],
    'recorded_decimal_precision': release['outcome_mean_policy']['outward_decimal_precision'],
    'admission_verified': analysis['admission_verified'],
    'reconstructed_intervals': {
        name: [str(value) for value in record['interval']]
        for name, record in analysis['methods'].items()},
}, indent=2))
