"""Assignment-only certification diagnostics and nested-count recovery.

Uses saved assignment observations; fresh draws occur only in the separately
seeded nested-count recovery experiment.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
import sys
sys.set_int_max_str_digits(0)
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from contextlib import contextmanager
from datetime import datetime,timezone
from fractions import Fraction as F
import fcntl
import gzip
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import resource
import shutil
import time
import traceback

from experiments import jmlr_verified_adaptive_v1_math as exact
from experiments.jmlr_verified_adaptive_v1_roots import RootPolicy,RootVerifier

PROJECT=Path(__file__).resolve().parents[1]
CONFIG=PROJECT/'configs/jmlr_efficiency_gap_v1.json'
ROOT=PROJECT/'results/jmlr_efficiency_gap_v1'
_PROTOCOL=None
_VERIFIERS={}


def now():return datetime.now(timezone.utc).isoformat()


def load(path):
    path=Path(path)
    if path.suffix=='.gz':
        with gzip.open(path,'rt') as f:return exact.decode(json.load(f))
    return exact.decode(json.loads(path.read_text()))


def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    if path.suffix=='.gz':
        with gzip.open(temp,'wt',compresslevel=3) as f:json.dump(exact.encode(value),f,separators=(',',':'))
    else:temp.write_text(json.dumps(exact.encode(value),indent=2)+'\n')
    os.replace(temp,path)






















