#!/usr/bin/env python3
"""Freeze a shared Jev-type specialist pilot for all decision model profiles."""
import argparse
import collections
import hashlib
import json
from pathlib import Path

from train_accuracy_lora import digest, option_specs, require, write_json

REVISION = 'c76749ec58bd8c3d2ea706b31c333a9059c38f90'
SEED = 20260927


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))


def distribution(value, ids):
    import math
    require(set(value) == set(ids), 'Probability labels differ from criteria')
    p = [float(value[k]) for k in ids]
    require(all(math.isfinite(v) and 0 <= v <= 1 for v in p), 'Invalid probability')
    require(abs(sum(p)-1) < 1e-4, 'Distribution must sum to one (rounding tolerance 1e-4)')
    return [v/sum(p) for v in p]


def decision(qid, q):
    criteria = q.get('criteria')
    kind = q['type']
    if kind == 'noul':
        criteria = criteria or {'false': 'No', 'true': 'Yes'}
        require(set(criteria) == {'false', 'true'}, 'Invalid Noul criteria')
        native = dict(type='binary', false_label=criteria['false'], true_label=criteria['true'])
    elif kind == 'choice':
        require(isinstance(criteria, dict) and 2 <= len(criteria) <= 26, 'Invalid Choice criteria')
        native = dict(type='choice', options=[dict(id=k, criterion=v) for k, v in criteria.items()])
    elif kind == 'score':
        require(isinstance(criteria, list) and 2 <= len(criteria) <= 10, 'Invalid Score criteria')
        native = dict(type='ordinal', levels=[dict(id=str(i), criterion=v, value=i)
                                             for i, v in enumerate(criteria)])
    else:
        raise ValueError('Unknown Jev type')
    require(isinstance(q['instructions'], str) and q['instructions'].strip(), 'Missing instruction')
    return dict(id=qid, instruction=q['instructions'], kind=native)


def convert(row):
    state, questions, gold = [json.loads(row[k]) for k in ('state', 'questions', 'gold')]
    require(questions and set(questions) == set(gold), 'Question/gold mismatch')
    decisions = [decision(k, q) for k, q in questions.items()]
    for d in decisions:
        g = gold[d['id']]
        ids = [o['id'] for o in option_specs(d)]
        distribution(g['probabilities'], ids)
        require(g['label'] in ids and g['type'] == questions[d['id']]['type'], 'Invalid gold')
    return dict(id=row['id'], workflow=row['workflow'], request=dict(state=state, decisions=decisions), gold=gold)


def state_key(case):
    return hashlib.sha256(canonical(case['request']['state']).encode()).hexdigest()


def check_disjoint(splits):
    ids, states = set(), set()
    for name, rows in splits.items():
        local_ids = {r['id'] for r in rows}
        local_states = {state_key(r) for r in rows}
        require(len(local_ids) == len(rows) == len(local_states), f'Duplicate ID/state in {name}')
        require(not (ids & local_ids or states & local_states), f'Cross-split ID/state leakage in {name}')
        ids.update(local_ids)
        states.update(local_states)


def jsonl(path, rows):
    with path.open('x') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False)+'\n')


def prepare(source, output):
    import pyarrow.parquet as pq
    train = [convert(r) for r in pq.read_table(source/'train.parquet').to_pylist()]
    test = [convert(r) for r in pq.read_table(source/'test.parquet').to_pylist()]
    require(len(train) == 1200 and len(test) == 400, 'Pinned split size changed')
    require(digest(source/'train.parquet') == '46a58d63edfd86e23229c78afe8b72307bb4ca9fb0e8df180cabb3c67ec9dcd5',
            'Pinned train source changed')
    require(digest(source/'test.parquet') == '4f294f218ea1da27f3efef936359389c62ea4d3973a41457732990f1d31b647c',
            'Pinned test source changed')
    check_disjoint(dict(source_train=train, test=test))
    grouped = collections.defaultdict(list)
    for row in train:
        grouped[row['workflow']].append(row)
    require(len(grouped) == 4 and all(len(v) == 300 for v in grouped.values()), 'Workflow allocation changed')
    splits = dict(train=[], development=[], unused=[], test=test)
    for workflow in sorted(grouped):
        rows = sorted(grouped[workflow], key=lambda r: hashlib.sha256(f'{SEED}:{r["id"]}'.encode()).hexdigest())
        splits['train'].extend(rows[:30])
        splits['development'].extend(rows[30:50])
        splits['unused'].extend(rows[50:])
    check_disjoint(splits)
    output.mkdir(parents=True, exist_ok=False)
    files = {}
    for name, rows in splits.items():
        labeled = output/f'{name}.jsonl'
        requests = output/f'{name}-requests.jsonl'
        jsonl(labeled, rows)
        jsonl(requests, [dict(id=r['id'], request=r['request']) for r in rows])
        files[name] = dict(cases=len(rows), decisions=sum(len(r['gold']) for r in rows),
                           ids=[r['id'] for r in rows], sha256=digest(labeled), requests_sha256=digest(requests))
    flat = [dict(id=f'{r["id"]}/{d["id"]}', request=dict(state=r['request']['state'], decisions=[d]))
            for r in splits['train'] for d in r['request']['decisions']]
    jsonl(output/'train-token-requests.jsonl', flat)
    write_json(output/'manifest.json', dict(schema_version=1, seed=SEED, mode='specialist',
        dataset='LocalLLaMA/typed-decisions', revision=REVISION,
        source_sha256={p.name: digest(p) for p in source.iterdir() if p.is_file()},
        splits=files, token_requests_sha256=digest(output/'train-token-requests.jsonl'),
        protocol=dict(epochs=1, rank=8, alpha=16, dropout=0, learning_rate=1e-4,
                      gradient_accumulation=12, mass_loss_weight=0.1,
                      objective='soft-target candidate cross entropy + candidate mass loss',
                      rotation='one SHA256-selected code rotation per decision, independent of labels',
                      selection='fixed final epoch; no development/test selection or calibration',
                      prompt_layout='legacy', prompt_detail='minimal'),
        scope='Synthetic teacher agreement; four seen workflows. Exact canonical state/ID disjointness only; no near-duplicate or pretraining exclusion.'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    prepare(a.source, a.output)
