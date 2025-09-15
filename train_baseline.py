"""train_baseline.py

Lightweight wrapper to run `rnntune.py` for a reproducible baseline.

Usage:
    python train_baseline.py --data-dir data --trials 20 --output-prefix baseline

This script calls `rnntune.py` in-place. It assumes `rnntune.py` reads `Xrn_prepared.csv` and `y1_prepared.csv` from the current working directory or the given data directory.
"""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', type=str, default='data', help='Directory with prepared CSVs')
    p.add_argument('--trials', type=int, default=20, help='Number of Optuna trials (reduced for baseline)')
    p.add_argument('--output-prefix', type=str, default='baseline', help='Prefix for saved models/metrics')
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    # Ensure required files exist
    xfile = data_dir / 'Xrn_prepared.csv'
    yfile = data_dir / 'y1_prepared.csv'
    if not xfile.exists() or not yfile.exists():
        print(f"Missing prepared data files in {data_dir}. Expected: Xrn_prepared.csv and y1_prepared.csv")
        sys.exit(1)

    # Call rnntune.py with environment variables so the script can pick up paths if modified to read them
    env = dict(**dict(**{}))
    env['RNNTUNE_X'] = str(xfile)
    env['RNNTUNE_Y'] = str(yfile)
    env['RNNTUNE_TRIALS'] = str(args.trials)
    env['RNNTUNE_OUTPUT_PREFIX'] = args.output_prefix

    cmd = [sys.executable, 'rnntune.py']
    print('Running:', ' '.join(cmd))

    try:
        subprocess.check_call(cmd, env={**env, **dict(**{})})
    except subprocess.CalledProcessError as e:
        print('rnntune.py failed with exit code', e.returncode)
        sys.exit(e.returncode)


if __name__ == '__main__':
    main()
