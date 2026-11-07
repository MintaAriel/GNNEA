"""Portable YAML configuration; paths are relative to the repository, never the caller's cwd."""
import argparse
from copy import deepcopy
import os
from pathlib import Path
import shutil
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / 'configs' / 'sirena.yml'


def merge(base, overrides):
    result = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def project_path(value):
    path = Path(os.path.expandvars(str(value))).expanduser()
    return path if path.is_absolute() else ROOT / path


def load_config(path=DEFAULT_CONFIG):
    path = project_path(path).resolve()
    base = ROOT / 'configs' / 'base.yaml'
    with base.open() as handle:
        config = yaml.safe_load(handle) or {}
    if path != base:
        with path.open() as handle:
            config = merge(config, yaml.safe_load(handle) or {})
    config['_config_path'] = str(path)
    return config


def config_from_cli():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config', default=str(DEFAULT_CONFIG))
    args, _ = parser.parse_known_args()
    return load_config(args.config)


def configure_runtime(config, activate_environment=True):
    """Select the configured physical GPU before torch is imported; nvtop uses this device numbering."""
    env_name = config['env']['conda_env']
    if activate_environment and Path(sys.prefix).name != env_name:
        conda = shutil.which('conda')
        if conda is None:
            raise RuntimeError(f'Activate the configured conda environment: {env_name}')
        os.execv(conda, [conda, 'run', '--no-capture-output', '-n', env_name,
                        'python', *sys.argv])
    runtime = config['runtime']
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(runtime['gpu'])
    os.environ['OMP_NUM_THREADS'] = str(runtime['threads'])
    os.environ['MKL_NUM_THREADS'] = str(runtime['threads'])
