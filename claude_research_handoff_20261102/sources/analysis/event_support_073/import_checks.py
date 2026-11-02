"""Independent import checks for completed 073; no inference or fitting."""
import hashlib
import json
import math
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'handoff/incoming/VIDOR-EVENT-SUPPORT-073'
P = DATA / 'remote_v1'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    return json.loads(p.read_text())


def main():
    assert sha(DATA / 'returned_v1.tar.gz') == '8c8316a98663ac92ffcdd68212182235f50380e162f042648b27c785760a0f83'
    plan = read(P / 'history_plan_v2/selection.json')
    seal = read(P / 'history_raw_v2/raw_seal.json')
    summary = read(P / 'history_evaluation_v1/summary.json')
    saved = {r['case_id']: r for r in read(P / 'history_evaluation_v1/per_case.json')}
    assert plan['complete'] is True and seal['complete'] is True
    assert seal['native_preprocessing_exact_all_batches'] is True
    assert seal['noop_exact'] is True and seal['model_state_unchanged'] is True
    assert seal['training_steps'] == 0
    assert seal['selection_sha256'] == sha(P / 'history_plan_v2/selection.json')
    for name in ['prepare_v2', 'infer_v2', 'evaluate', 'eval_tests']:
        assert read(P / (name + '_completion.json'))['returncode'] == 0
    for name in ['history_probe.py', 'gt_input_source.py']:
        assert sha(ROOT / 'analysis/event_support_073' / name) == sha(P / 'code_v2' / name)
    assert sha(ROOT / 'analysis/event_support_073/analyze_history.py') == sha(P / 'code_eval_v1/analyze_history.py')
    eval_sources = read(P / 'history_evaluation_v1/source_hashes.json')
    assert eval_sources['script_sha256'] == sha(P / 'code_eval_v1/analyze_history.py')
    assert eval_sources['seal']['sha256'] == sha(P / 'history_raw_v2/raw_seal.json')
    assert eval_sources['input_audits']['sha256'] == sha(P / 'history_raw_v2/input_audits.json')
    cases = plan['cases']
    identities = [(r['video_name'], r['subject_tid'], r['object_tid'], r['predicate'],
                   tuple(r['previous']), tuple(r['next'])) for r in cases]
    assert len(set(identities)) == len(cases)
    expected = {f"{r['case_id']}_{a}.npz" for r in cases for a in ['B', 'N', 'P', 'R']}
    assert set(seal['raw_sha256']) == expected
    assert {p.name for p in (P / 'history_raw_v2/raw').glob('*.npz')} == expected
    for n, h in seal['raw_sha256'].items():
        assert sha(P / 'history_raw_v2/raw' / n) == h
    audits = read(P / 'history_raw_v2/input_audits.json')
    assert len(audits) == len(expected)
    assert len({(a['case_id'], a['arm']) for a in audits}) == len(expected)
    for a in audits:
        assert all(a[k] is True for k in ['future_exact', 'gap_exact', 'next_exact', 'length_unchanged'])
    raw_effects = []
    for r in cases:
        assert r['Tpad'] == max(512, math.ceil(r['N'] / 64) * 64)
        assert r['selection_uses_model_outputs'] is False
        l, u = r['edit']
        assert u - l == 8
        assert r['origin'] + 4 * u <= r['previous'][1] - max(4, math.ceil(.5 * r['fps']))
        for arm in ['N', 'P']:
            assert r['donors'][arm]['start'] + 8 <= l
        assert r['origin'] + 4 * (r['donors']['N']['start'] + 8) <= r['previous'][0] - max(4, math.ceil(.5 * r['fps']))
        assert .5 <= r['donors']['RMS_ratio_N_over_P'] <= 2
        scores = {}
        for arm in ['B', 'N', 'P', 'R']:
            with np.load(P / 'history_raw_v2/raw' / f"{r['case_id']}_{arm}.npz") as a:
                logits = a['pred_logits'].astype('float64')
                probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
                probabilities /= probabilities.sum(axis=1, keepdims=True)
                temporal = 1 / (1 + np.exp(-a['pred_masks'].astype('float64')))
                value = (probabilities[:, r['predicate_id'], None] * temporal).max(axis=0)
            scores[arm] = {region: float(value[r[region + '_indices']].mean()) for region in ['gap', 'next']}
            for region in ['gap', 'next']:
                assert abs(saved[r['case_id']]['arms'][arm][region + '_mean'] - scores[arm][region]) < 1e-12
        raw_effects.append({'video': r['video_name'], 'off': scores['P']['gap'] - scores['N']['gap'],
                            'restart': scores['N']['next'] - scores['P']['next'], 'short': r['next_short']})
    independent = {}
    for field in ['off', 'restart']:
        grouped = defaultdict(list)
        for row in raw_effects:
            grouped[row['video']].append(row[field])
        means = np.array([np.mean(grouped[v]) for v in sorted(grouped)])
        # Independent raw-output calculation; use frozen resampling seed/definition.
        rng = np.random.default_rng(73073)
        resampled = np.array([means[rng.choice(len(means), len(means), replace=True)].mean() for _ in range(10000)])
        bounds = np.quantile(resampled, [.0125, .9875])
        expected_result = summary['primary_all_cases']['primary'][field]
        assert abs(means.mean() - expected_result['effect_P_vs_N']['point']) < 1e-12
        assert np.allclose(bounds, expected_result['effect_P_vs_N']['CI_97_5_percent'], rtol=0, atol=1e-12)
        assert bounds[1] < .03
        assert expected_result['decision'] == 'BELOW_PRESPECIFIED_EFFECT'
        independent[field] = {'point': means.mean(), 'interval': bounds.tolist()}
    result = {'status': 'PASS', 'N_cases': len(cases), 'N_videos': len({r['video_name'] for r in cases}),
              'N_short': sum(r['next_short'] for r in cases), 'N_raw_files': len(expected),
              'N_unique_physical_events': len(set(identities)), 'all_raw_hashes': 'PASS',
              'native_preprocessing_exact_all_batches': True, 'padding_rule': 'PASS',
              'temporal_donor_bounds': 'PASS', 'independent_primary_from_raw': independent,
              'predicate_counts': dict(Counter(r['predicate'] for r in cases)),
              'script_sha256': sha(Path(__file__)), 'bundle_sha256': sha(DATA / 'returned_v1.tar.gz'),
              'scope': 'Independent numeric/import checks, no visual semantic audit, no generalization claim'}
    (DATA / 'IMPORT_CHECKS.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
