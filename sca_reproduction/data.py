"""Download and validate the two documented public data inputs."""
from pathlib import Path
import hashlib
import urllib.request

from experiments.jmlr_release_capacity_v1_sampling import SOURCE_CSV, SOURCE_SHA256, build_benchmark
from experiments.jmlr_empirical_relevance_v1_core import RHC_SHA256, load_rhc


RHC_CSV = 'https://hbiostat.org/data/repo/rhc.csv'


def _download(path, url, checksum):
    if not path.exists():
        request = urllib.request.Request(url, headers={'User-Agent': 'SCA-code-reproduction/0.1'})
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = response.read()
        if hashlib.sha256(payload).hexdigest() != checksum:
            raise ValueError('Public download checksum differs; input was not installed')
        temporary = path.with_suffix(path.suffix+'.download')
        temporary.write_bytes(payload)
        temporary.replace(path)


def prepare_data(directory='data', dataset='all'):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if dataset not in ('all', 'ihdp', 'rhc'):
        raise ValueError('dataset must be all, ihdp, or rhc')
    result = {}
    if dataset in ('all', 'ihdp'):
        path = directory/'ihdp_npci_1.csv'; _download(path, SOURCE_CSV, SOURCE_SHA256)
        benchmark = build_benchmark(path)
        result['ihdp'] = {'path': str(path.resolve()), 'sha256': SOURCE_SHA256,
                          'reference_records': len(benchmark['record_states']),
                          'states': benchmark['law']['K'], 'source': SOURCE_CSV,
                          'scope': 'covariate-only semi-synthetic benchmark input'}
    if dataset in ('all', 'rhc'):
        path = directory/'rhc.csv'; _download(path, RHC_CSV, RHC_SHA256)
        rhc = load_rhc(path, RHC_SHA256)
        result['rhc'] = {'path': str(path.resolve()), 'sha256': RHC_SHA256,
                         'records': rhc['rows'], 'states': len(rhc['state_counts']),
                         'source': RHC_CSV,
                         'scope': 'observed treatment/outcome illustration'}
    return result
