#!/usr/bin/env python3
"""Fetch pinned checkpoints or run the same Jev LoRA pipeline for one/all models.

No downloads occur in run mode. Each model has its own output, hashes, smoke,
adapter, conversion, paired native evaluation and failure record. GPUs run serially.
"""
import argparse
import datetime
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from jev_model_profiles import DEFAULT_PROFILES, read_profile
from train_accuracy_lora import digest, write_json


def seal_tokens(data, tokens, model, exporter):
    write_json(tokens.with_suffix('.seal.json'), dict(schema_version=1, split='train',
        manifest_sha256=digest(data/'manifest.json'), tokens_sha256=digest(tokens),
        requests_sha256=digest(data/'train-token-requests.jsonl'), model_sha256=digest(model),
        exporter_sha256=digest(exporter), prompt_layout='legacy', prompt_detail='minimal'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['fetch', 'run'])
    p.add_argument('--models', nargs='+', default=['gemma4'], help='Profile names, or all')
    p.add_argument('--profiles', type=Path, default=DEFAULT_PROFILES)
    p.add_argument('--checkpoint-root', type=Path, required=True, help='Contains pinned HF revision directories')
    p.add_argument('--data', type=Path)
    p.add_argument('--gguf-root', type=Path, default=Path('models'))
    p.add_argument('--output', type=Path, required=True, help='New run directory; never overwritten')
    p.add_argument('--exporter', type=Path, default=Path('target/release/examples/export_decision_tokens'))
    p.add_argument('--evaluator', type=Path, default=Path('target/release/examples/evaluate_jsonl'))
    p.add_argument('--converter', type=Path, help='Matching llama.cpp convert_lora_to_gguf.py')
    p.add_argument('--library-path', type=Path, help='Matching llama.cpp shared libraries')
    p.add_argument('--rules-fixture', type=Path, help='Optional existing decision-rules regression fixture')
    a = p.parse_args()
    names = list(json.loads(a.profiles.read_text())['models']) if a.models == ['all'] else a.models
    if not names or any(re.fullmatch(r'[a-z0-9][a-z0-9_-]*', n) is None for n in names):
        p.error('Model profile keys must be safe lowercase names without path separators')
    profiles = [read_profile(n, a.profiles) for n in names]
    if len(set(names)) != len(names):
        p.error('Model names must be unique')
    if a.command == 'run' and (a.data is None or a.converter is None):
        p.error('run requires --data and --converter')
    a.output.mkdir(parents=True, exist_ok=False)
    scripts = Path(__file__).resolve().parent
    env = dict(os.environ)
    if a.library_path:
        library_variable = 'PATH' if os.name == 'nt' else 'LD_LIBRARY_PATH'
        env[library_variable] = str(a.library_path.resolve()) + (os.pathsep+env[library_variable] if env.get(library_variable) else '')
    statuses = []
    for profile in profiles:
        out = a.output/profile['name']
        out.mkdir()
        checkpoint = (a.checkpoint_root/profile['revision']).resolve()
        status = dict(model=profile['name'], profile=profile, started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), stages=[])
        statuses.append(status)

        def stage(name, cmd):
            cmd = [str(x) for x in cmd]
            started = time.monotonic()
            with (out/f'{name}.log').open('x') as log:
                result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
            status['stages'].append(dict(name=name, command=cmd, exit_code=result.returncode,
                                        elapsed_s=time.monotonic()-started))
            print(profile['name'], name, 'exit', result.returncode, flush=True)
            if result.returncode:
                raise RuntimeError(f'{name} failed; see {out/name}.log')

        try:
            if a.command == 'fetch':
                from huggingface_hub import snapshot_download
                snapshot_download(profile['hf_model'], revision=profile['revision'], local_dir=checkpoint,
                    allow_patterns=['*.json', '*.safetensors', '*.jinja', '*.model', 'LICENSE*', 'NOTICE*'])
                status['checkpoint'] = str(checkpoint)
            else:
                model = (a.gguf_root/profile['gguf']).resolve()
                for path in (checkpoint/'config.json', model, a.exporter, a.evaluator, a.converter):
                    if not path.is_file():
                        raise FileNotFoundError(f'Missing local prerequisite: {path}')
                status['artifacts'] = {k:digest(v) for k,v in dict(model=model, exporter=a.exporter,
                    evaluator=a.evaluator, converter=a.converter, manifest=a.data/'manifest.json').items()}
                tokens = out/'train-tokens.jsonl'
                stage('export', [a.exporter.resolve(), '--model', model, '--input', a.data/'train-token-requests.jsonl',
                      '--output', tokens, '--all-rotations', '--prompt-layout', 'legacy', '--prompt-detail', 'minimal'])
                seal_tokens(a.data, tokens, model, a.exporter)
                shared = ['--model', profile['name'], '--profiles', a.profiles, '--data', a.data,
                          '--tokens', tokens, '--checkpoint', checkpoint]
                stage('smoke', [sys.executable, scripts/'train_jev_lora.py', 'smoke', *shared, '--output', out/'smoke'])
                stage('train', [sys.executable, scripts/'train_jev_lora.py', 'train', *shared, '--output', out/'train',
                               '--smoke-report', out/'smoke/complete.json'])
                stage('convert', [sys.executable, a.converter.resolve(), out/'train/adapter', '--base', checkpoint,
                                  '--outfile', out/'adapter.gguf', '--outtype', 'f16'])
                status['artifacts']['adapter'] = digest(out/'adapter.gguf')
                if a.rules_fixture:
                    stage('prepare-regression', [sys.executable, scripts/'report_jev_rule_regression.py',
                        '--fixture', a.rules_fixture, '--requests-output', out/'rules-requests.jsonl'])
                for variant in ('base', 'adapter'):
                    pred = out/f'{variant}-predictions.jsonl'
                    cmd = [a.evaluator.resolve(), '--model', model, '--input', a.data/'test-requests.jsonl', '--output', pred,
                           '--cuda', '--context', '8192', '--batch', '256', '--threads', '4', '--execution-mode', 'fresh',
                           '--prompt-layout', 'legacy', '--prompt-detail', 'minimal', '--request-batch-size', '1',
                           '--model-load-mode', 'read', '--warmup']
                    if variant == 'adapter':
                        cmd += ['--lora', out/'adapter.gguf']
                    stage(variant, cmd)
                    stage(variant+'-report', [sys.executable, scripts/'report_jev.py', '--data', a.data,
                        '--predictions', pred, '--output', out/(variant+'-report')])
                    summary = json.loads((out/(variant+'-report')/'summary.json').read_text())
                    status.setdefault('metrics', {})[variant] = dict(summary['overall'], latency_ms=summary['latency_ms'],
                                                                    by_type=summary['by_type'])
                    if summary['failures']:
                        raise RuntimeError(f'{variant}: incomplete evaluation; failures retained in report')
                    if a.rules_fixture:
                        regression = out/f'{variant}-rules-predictions.jsonl'
                        cmd[cmd.index('--input')+1] = out/'rules-requests.jsonl'
                        cmd[cmd.index('--output')+1] = regression
                        stage(variant+'-rules', cmd)
                        stage(variant+'-rules-report', [sys.executable, scripts/'report_jev_rule_regression.py',
                            '--fixture', a.rules_fixture, '--predictions', regression,
                            '--output', out/f'{variant}-rules-report.json'])
                status['delta_after_minus_before'] = {k: status['metrics']['adapter'][k]-status['metrics']['base'][k]
                    for k in ('raw_accuracy', 'coverage', 'accepted_accuracy', 'correct_all', 'soft_kl', 'soft_brier', 'score_mae')
                    if status['metrics']['adapter'][k] is not None and status['metrics']['base'][k] is not None}
            status['status'] = 'ok'
        except Exception as error:
            status.update(status='failed', error=str(error))
            print(profile['name'], 'failed:', error, flush=True)
        write_json(out/'run.json', status)
    write_json(a.output/'summary.json', dict(schema_version=1, command=a.command, runs=statuses,
        scope='ok means pipeline completed, not accuracy improved. Only the requested models were run.'))
    if any(s['status'] != 'ok' for s in statuses):
        sys.exit(1)


if __name__ == '__main__':
    main()
