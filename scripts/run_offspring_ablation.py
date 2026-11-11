"""Run the matched offspring ablation configured entirely in configs/sirena.yml."""
import argparse
from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import config_from_cli, configure_runtime, project_path


def main():
    config = config_from_cli()
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--gpu', type=int)
    gpu = pre.parse_known_args()[0].gpu
    if gpu is not None:                                       # one process per GPU (see --shard)
        config['runtime']['gpu'] = gpu
    configure_runtime(config)
    from src.training import fingerprint, run_training
    from src.analysis import analyze
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=config['_config_path'])
    parser.add_argument('--analyze-only', action='store_true')
    parser.add_argument('--gpu', type=int, help='physical GPU for this process (overrides runtime.gpu)')
    parser.add_argument('--prepare-only', action='store_true',
                        help='build the MXtal cluster cache (parents + children) once and report Z\' counts')
    parser.add_argument('--shard', help='I/N: train only jobs with index %% N == I and skip the analysis')
    args = parser.parse_args()
    grid = config['experiments']
    if args.prepare_only:
        from collections import Counter
        from src.training import prepare_data
        cfg = deepcopy(config)
        cfg['training']['offspring_structure'] = True
        _, _, cache = prepare_data(cfg, 'mxtal')
        for state in ('relaxed', 'unrelaxed'):
            counts = Counter(int(c.get('z_prime', c['n_mol'])) for k, c in cache.items() if k[0] == state)
            print(f"inside={cfg['mxtal'].get('inside', 'cell')}: {state} structures by Z' (molecules inside): "
                  f'{dict(sorted(counts.items()))}', flush=True)
        return
    out = project_path(grid['output_dir'])
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for encoder in grid['encoders']:
        for fusion in grid['fusions']:
            for split, seeds in grid['splits'].items():
                for seed in seeds:
                    for energy in grid.get('energy_features', [config['training']['energy_features']]):
                        for child in grid['offspring_structure']:
                            cfg = deepcopy(config)
                            cfg['training'].update(fusion=fusion, split=split, seed=seed, offspring_structure=child,
                                                   energy_features=energy)
                            name = (f"{encoder}_{fusion}_{split}_s{seed}{'_energy' if energy else ''}_"
                                    f"{'offspring' if child else 'parents'}")
                            jobs.append((name, encoder, cfg))
    shard = tuple(map(int, args.shard.split('/'))) if args.shard else None
    for index, (name, encoder, cfg) in enumerate(jobs, 1):
        if shard and (index - 1) % shard[1] != shard[0]:
            continue
        run_out = out / 'runs' / name
        metric_path = run_out / 'metrics.json'
        if metric_path.exists():
            previous = json.loads(metric_path.read_text())
            if previous.get('fingerprint') != fingerprint(cfg, encoder):
                raise RuntimeError(f'{name}: existing results have different data/config/code; select a new output_dir.')
            if grid['resume'] or args.analyze_only:
                print(f'[{index}/{len(jobs)}] verified {name}', flush=True)
                continue
        if args.analyze_only:
            raise FileNotFoundError(f'Incomplete experiment: {metric_path}')
        run_out.mkdir(parents=True, exist_ok=True)
        print(f'[{index}/{len(jobs)}] training {name}', flush=True)
        with (run_out / 'train.log').open('w', buffering=1) as log:
            with redirect_stdout(log), redirect_stderr(log):
                run_training(cfg, encoder, run_out)
    if shard:
        print(f'Shard {args.shard} finished.', flush=True)
        return
    analyze(config, [out / 'runs' / name for name, _, _ in jobs])
    print(f'Completed {len(jobs)} runs. Results: {out}', flush=True)


if __name__ == '__main__':
    main()
