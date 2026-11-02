"""Second reconstruction of075 saved intervals and official detection ranks.

Does not import075 training/evaluation code. This is a computational audit,
not an independent scientific review or a fresh generalization experiment.
"""
import argparse
import csv
import gzip
import hashlib
import json
import time
from pathlib import Path

import numpy as np


def load(path):
    path = Path(path)
    with (gzip.open(path, 'rt') if path.suffix == '.gz' else path.open()) as f:
        return json.load(f)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def verify(root):
    start = time.monotonic()
    out, code = root / 'results_v2', root / 'code_v2'
    summary = load(out / 'summary.json')
    assert summary['complete'] is True
    contracts = load(code / 'cohorts.json')
    expected = {(r['cohort'], r['video_name'], r['gt_index']): r
                for r in load(out / 'per_gt.json.gz')}
    expected_videos = {(r['cohort'], r['video_name']): r
                       for r in load(out / 'per_video.json')}
    arms = ('baseline', 'original', 'balanced', 'ordered')
    for name, digest in load(code / 'software_freeze.json').items():
        assert sha(code / name) == digest, name
    seal = load(out / 'intervals_seal_before_GT.json')
    expected_interval_paths = {
        'intervals/' + cohort + '/' + video + '.json.gz'
        for cohort, c in contracts.items()
        for video in load(Path(c['export']) / 'manifest.json')['video_names']
    }
    assert len(seal) == 128 and set(seal) == expected_interval_paths
    assert {str(p.relative_to(out)) for p in (out/'intervals').rglob('*.json.gz')} == set(seal)
    for name, digest in seal.items():
        assert sha(out / name) == digest, name
    checks = dict(videos=0, GT=0, candidate_identity_checks=0,
                  prediction_hashes=0, official_rank_values=0)
    independently_recomputed = {}
    for cohort, contract in contracts.items():
        export = Path(contract['export'])
        manifest = load(export / 'manifest.json')
        assert sha(export / 'manifest.json') == contract['manifest_sha256']
        official = export / 'official.py'
        assert sha(official) == manifest['outputs']['official.py']['sha256']
        namespace = {}
        exec(compile(official.read_text(), str(official), 'exec'), namespace)
        ranks_all = {arm: [] for arm in arms}
        aps_all = {arm: [] for arm in arms}
        for video in manifest['video_names']:
            def source(name):
                path = export / 'videos' / video / name
                assert sha(path) == manifest['outputs'][str(path.relative_to(export))]['sha256']
                return load(path)
            geometry = source('geometry.json.gz')
            baseline = source('baseline.json.gz')
            selected = source('selected.json.gz')['selected']
            gt = source('gt.json.gz')
            intervals = load(out / 'intervals' / cohort / (video + '.json.gz'))
            assert intervals['N_candidates'] == len(baseline) == len(selected)
            assert intervals['video_name'] == video
            for arm in arms:
                ds = intervals['durations'][arm]
                assert len(ds) == len(baseline)
                predictions, shared_query = [], {}
                for original, candidate, duration in zip(baseline, selected, ds):
                    begin, end = duration
                    assert type(begin) is int and type(end) is int and end - begin >= 5
                    pair = geometry['pairs'][candidate['pair_idx']]
                    assert candidate['sid'] == pair['sid'] and candidate['oid'] == pair['oid']
                    origin = pair['pair_start'] + pair['offset']
                    assert (begin - origin) % 4 == 0 and (end - 1 - origin) % 4 == 0
                    assert pair['pair_start'] <= begin < end <= pair['pair_end']
                    query_key = (candidate['pair_idx'], candidate['query'])
                    if query_key in shared_query:
                        assert shared_query[query_key] == duration, (video, arm, 'query time disagreement')
                    shared_query[query_key] = duration
                    tracks = []
                    for tid in (candidate['sid'], candidate['oid']):
                        first, last = geometry['durations'][tid]
                        assert first <= begin < end <= last
                        tracks.append([geometry['boxes'][tid][frame - first]
                                       for frame in range(begin, end)])
                    prediction = dict(triplet=original['triplet'], score=original['score'],
                                      duration=duration, sub_traj=tracks[0], obj_traj=tracks[1])
                    assert set(prediction) == set(original)
                    predictions.append(prediction)
                    checks['candidate_identity_checks'] += 1
                assert canonical(predictions) == intervals['canonical_prediction_sha256'][arm]
                if arm == 'baseline':
                    assert predictions == baseline
                checks['prediction_hashes'] += 1
                precision, recall, _, ranks = namespace['eval_detection_scores_v2'](gt, predictions, .5)
                ap = float(namespace['voc_ap'](recall, precision))
                assert ap == expected_videos[(cohort, video)]['AP'][arm]
                if len(gt):
                    aps_all[arm].append(ap)
                for gt_index, rank in enumerate(ranks):
                    assert rank == expected[(cohort, video, gt_index)]['ranks'][arm]
                    checks['official_rank_values'] += 1
                ranks_all[arm].extend(ranks.tolist())
            checks['videos'] += 1
            checks['GT'] += len(gt)
        own = {}
        for arm in arms:
            hits = {str(k): sum(0 <= x < k for x in ranks_all[arm]) for k in (50, 100, 200)}
            changes = {}
            for control in arms:
                changes[control] = {}
                for k in (50, 100, 200):
                    new = sum(0 <= a < k and not 0 <= b < k
                              for a, b in zip(ranks_all[arm], ranks_all[control]))
                    lost = sum(0 <= b < k and not 0 <= a < k
                               for a, b in zip(ranks_all[arm], ranks_all[control]))
                    changes[control][str(k)] = dict(new=new, lost=lost, net=new-lost)
            ap = float(np.mean(aps_all[arm]))
            assert ap == summary['cohorts'][cohort]['mAP'][arm]
            reported = summary['cohorts'][cohort]['groups']['all']['arms'][arm]
            assert hits == reported['hits'] and changes == reported['comparisons']
            own[arm] = dict(mAP=ap, hits=hits, comparisons=changes)
        independently_recomputed[cohort] = own
        print('VERIFIED075', cohort, flush=True)
    assert checks['videos'] == 128 and checks['GT'] == 4739
    all_rows = list(expected.values())
    membership = {
        'all': lambda r: True,
        'short': lambda r: r['relation_sec'] < 3,
        'subsecond': lambda r: r['relation_sec'] < 1,
        'long': lambda r: r['relation_sec'] >= 3,
        'C': lambda r: r['original_group'] == 'C',
        'short_C': lambda r: r['relation_sec'] < 3 and r['original_group'] == 'C',
        'F': lambda r: r['input_feasible'],
        'short_C_F': lambda r: r['input_feasible'] and r['relation_sec'] < 3 and r['original_group'] == 'C',
        'long_coexist_short': lambda r: r['relation_sec'] < 3 and r['joint_local_continuous_sec'] >= 10,
        'long_coexist_short_C': lambda r: r['relation_sec'] < 3 and r['joint_local_continuous_sec'] >= 10 and r['original_group'] == 'C',
        'long_coexist_persistent': lambda r: r['relation_sec'] >= 3 and r['joint_local_continuous_sec'] >= 10,
        'fixed073_grid_not_threshold': lambda r: r['fixed073_grid_not_threshold'],
        'T': lambda r: r['T'], 'U': lambda r: r['U'], 'V': lambda r: r['V'],
    }
    fixed = [r for r in all_rows if r['fixed073_grid_not_threshold']]
    assert len(fixed) == 63 and all(r['original_group'] == 'C' for r in fixed)
    for cohort in (*contracts, 'combined'):
        cohort_rows = all_rows if cohort == 'combined' else [r for r in all_rows if r['cohort'] == cohort]
        for name, belongs in membership.items():
            members = [r for r in cohort_rows if belongs(r)]
            group = summary['cohorts'][cohort]['groups'][name]
            assert group['N_GT'] == len(members)
            assert group['N_videos'] == len({r['video_name'] for r in members})
            for arm in arms:
                for k in (50, 100, 200):
                    assert group['arms'][arm]['hits'][str(k)] == sum(0 <= r['ranks'][arm] < k for r in members)
                    for control in arms:
                        new = sum(0 <= r['ranks'][arm] < k and not 0 <= r['ranks'][control] < k for r in members)
                        lost = sum(0 <= r['ranks'][control] < k and not 0 <= r['ranks'][arm] < k for r in members)
                        assert group['arms'][arm]['comparisons'][control][str(k)] == dict(new=new, lost=lost, net=new-lost)
    def hit(cohort, group, arm):
        return summary['cohorts'][cohort]['groups'][group]['arms'][arm]['hits']['100']
    controls = ('baseline', 'original', 'balanced')
    gates = {
        'overall_both_cohorts': all(summary['cohorts'][c]['mAP']['ordered'] >= summary['cohorts'][c]['mAP'][a]
                                  and hit(c, 'all', 'ordered') >= hit(c, 'all', a)
                                  for c in contracts for a in controls),
        'combined_target_exceeds_all_controls': all(hit('combined', 'long_coexist_short', 'ordered') > hit('combined', 'long_coexist_short', a) for a in controls),
        'target_no_net_loss_each_cohort': all(hit(c, 'long_coexist_short', 'ordered') >= hit(c, 'long_coexist_short', 'baseline') for c in contracts),
        'fixed_073_grid_not_threshold_actual_recovery': summary['cohorts']['combined']['groups']['fixed073_grid_not_threshold']['arms']['ordered']['comparisons']['baseline']['100']['new'] >= 1,
        'persistent_no_net_loss': all(hit(c, g, 'ordered') >= hit(c, g, 'baseline') for c in contracts for g in ('long', 'long_coexist_persistent')),
    }
    assert gates == summary['gates']
    assert summary['decision'] == ('SMALL_PILOT_CONTINUATION_SIGNAL' if all(gates.values()) else 'DO_NOT_EXPAND_THIS_VERSION')
    result = dict(complete=True, checks=checks, cohorts=independently_recomputed,
                  summary_sha256=sha(out/'summary.json'),
                  per_gt_sha256=sha(out/'per_gt.json.gz'), elapsed_seconds=time.monotonic()-start,
                  scope='Independent trajectory reconstruction and official matching; no training',
                  scientific_review='SEPARATE_REQUIRED')
    destination = root / 'verification_v2'
    destination.mkdir(exist_ok=False)
    (destination/'IMPORT_CHECKS.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(dict(complete=True, checks=checks)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    verify(parser.parse_args().root)
