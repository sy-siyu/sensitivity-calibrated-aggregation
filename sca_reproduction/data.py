"""Download and validate the documented public IHDP covariate input."""
from pathlib import Path
import hashlib
import urllib.request

from experiments.jmlr_release_capacity_v1_sampling import SOURCE_CSV, SOURCE_SHA256, build_benchmark


def prepare_data(directory='data'):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'ihdp_npci_1.csv'
    if not path.exists():
        request = urllib.request.Request(SOURCE_CSV, headers={'User-Agent': 'SCA-code-reproduction/0.1'})
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = response.read()
        if hashlib.sha256(payload).hexdigest() != SOURCE_SHA256:
            raise ValueError('Public download checksum differs; input was not installed')
        temporary = path.with_suffix('.download')
        temporary.write_bytes(payload)
        temporary.replace(path)
    # Checks checksum, dimensions, variable construction and complete support.
    benchmark = build_benchmark(path)
    return {'path': str(path.resolve()), 'sha256': SOURCE_SHA256,
            'reference_records': len(benchmark['record_states']),
            'states': benchmark['law']['K'], 'source': SOURCE_CSV}
