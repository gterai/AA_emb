#!/usr/bin/env python3
"""Run the published primary, protein-cluster or leave-one-function-out experiments."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import threading
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tests'))
from check_data import validate_assets

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT
GROUPS = {
    'primary': ['c80'],
    'protein': ['c50', 'c70', 'c90'],
    'function': ['GO_0005840', 'GO_0005886', 'GO_0005576', 'GO_0005739', 'GO_0002376', 'GO_0003723'],
}
ARGS = {'mrna_only': [], 'mrna_t5u': ['--emb_name', 'emb_T5u'],
        't5u_only': ['--emb_name', 'emb_T5u', '--abl_type', 'm']}


PRIMARY_ARGS = {'mrna_only': ['--emb_name', 'emb_T5u', '--abl_type', 'p']}
for name, key in [('t5b','emb_T5b'), ('t5u','emb_T5u'), ('ank','emb_ank'),
                  ('ank3','emb_ank3'), ('esm2','emb_esm2'), ('esm2l','emb_esm2L')]:
    PRIMARY_ARGS[name + '_only'] = ['--emb_name', key, '--abl_type', 'm']
    PRIMARY_ARGS['mrna_' + name] = ['--emb_name', key]
PRIMARY_ARGS.update({'mrna_aacom': ['--emb_name', 'emb_aacom'],
                     'mrna_dipep': ['--emb_name', 'emb_dipep']})


def run_with_live_logs(command, run, environment):
    """Stream both child outputs to the terminal and their separate log files."""
    environment = dict(environment, PYTHONUNBUFFERED='1')

    def forward(source, saved, terminal):
        for line in source:
            saved.write(line)
            saved.flush()
            terminal.write(line)
            terminal.flush()

    with (run / 'stdout.txt').open('w') as stdout, (run / 'stderr.txt').open('w') as stderr:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True, bufsize=1, env=environment) as process:
            readers = [threading.Thread(target=forward, args=(source, saved, terminal))
                       for source, saved, terminal in
                       [(process.stdout, stdout, sys.stdout),
                        (process.stderr, stderr, sys.stderr)]]
            for reader in readers:
                reader.start()
            try:
                returncode = process.wait()
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                for reader in readers:
                    reader.join()
                process.stdout.close()
                process.stderr.close()
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('analysis', choices=GROUPS)
    p.add_argument('--input', type=Path, default=REPO / 'data/processed/input.pkl.gz')
    p.add_argument('--output', type=Path, help='Fresh output directory; existing run directories are rejected')
    p.add_argument('--groups', nargs='+')
    p.add_argument('--conditions', nargs='+', choices=sorted(set(ARGS) | set(PRIMARY_ARGS)))
    p.add_argument('--seeds', nargs='+', type=int, default=list(range(10)))
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    p.add_argument('--save-checkpoints', action='store_true', help='Save locally only (ignored by Git)')
    p.add_argument('--dry-run', action='store_true', help='Validate partitions and print commands without loading input or training')
    a = p.parse_args()
    groups = a.groups or GROUPS[a.analysis]
    model_args = PRIMARY_ARGS if a.analysis == 'primary' else ARGS
    conditions = a.conditions or (list(model_args) if a.analysis != 'function' else ['mrna_only', 'mrna_t5u'])
    if not set(groups) <= set(GROUPS[a.analysis]) or len(groups) != len(set(groups)):
        p.error('Invalid or duplicate groups')
    if not set(conditions) <= set(model_args) or len(conditions) != len(set(conditions)) or (a.analysis == 'function' and 't5u_only' in conditions):
        p.error('Invalid or duplicate conditions for this analysis')
    if not set(a.seeds) <= set(range(10)) or len(a.seeds) != len(set(a.seeds)) or a.epochs < 1:
        p.error('Seeds must be distinct values in 0..9 and epochs must be positive')
    print("Checking published partitions and data...", flush=True)
    validate_assets()
    name = {'primary': 'main_comparison', 'protein': 'protein_cluster_baseline', 'function': 'go_slim_holdout'}[a.analysis]
    output = (a.output or ROOT / 'training/runs' / name).resolve()
    commands = []
    for condition in conditions:
        for group in groups:
            for seed in a.seeds:
                run = (output / condition / f'seed_{seed}' if a.analysis == 'primary'
                       else output / condition / group / f'seed_{seed}')
                if run.exists():
                    p.error(f'Run already exists: {run}. Choose a fresh --output or unrun conditions/seeds.')
                if a.analysis == 'primary':
                    partition = ROOT / 'partitions/mrna_c80'
                elif a.analysis == 'protein':
                    partition = ROOT / 'partitions' / f'protein_{group}'
                else:
                    partition = ROOT / 'partitions/function_holdout' / group
                split = partition / f'seed_{seed}' / 'class.txt'
                command = [sys.executable, str(REPO / 'training/eval_multiemb.py'), str(a.input.resolve()),
                           *model_args[condition], '--input_class_fname', str(split), '--seed', str(seed),
                           '--epoch', str(a.epochs), '--lr', '0.0001', '--s_bat', '100', '--device', a.device,
                           '--out_class_fname', str(run / 'class_used.txt'),
                           '--model_fname', str(run / 'model.pth') if a.save_checkpoints else '',
                           '--metrics_tsv', str(run / 'metrics.tsv')]
                commands.append((run, command))
    print(f'{len(commands)} runs; output: {output}', flush=True)
    if a.dry_run:
        for _, command in commands:
            print(shlex.join(command))
        return
    # Validate in a separate process so the large input is released before training.
    print(f'Checking input: {a.input.resolve()} (loading the full file may take time)...', flush=True)
    subprocess.run([sys.executable, '-u', str(ROOT / 'tests/check_data.py'), '--input', str(a.input.resolve())], check=True)
    print('Input check complete.', flush=True)
    environment = dict(os.environ)
    environment.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    for index, (run, command) in enumerate(commands, 1):
        run.mkdir(parents=True, exist_ok=False)
        (run / 'command.json').write_text(json.dumps(command, indent=2) + '\n')
        print(f'[{index}/{len(commands)}] Starting training: {run}', flush=True)
        print(f'Live logs are also saved to {run / "stdout.txt"} and {run / "stderr.txt"}', flush=True)
        try:
            run_with_live_logs(command, run, environment)
        except subprocess.CalledProcessError as error:
            print(f'Training failed (exit {error.returncode}). See {run / "stderr.txt"}',
                  file=sys.stderr, flush=True)
            raise
        print(f'[{index}/{len(commands)}] Completed: {run / "metrics.tsv"}', flush=True)



if __name__ == '__main__':
    main()
