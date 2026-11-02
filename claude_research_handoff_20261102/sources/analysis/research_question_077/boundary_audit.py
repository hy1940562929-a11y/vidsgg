#!/usr/bin/env python3
"""Frozen GT27 mask-boundary diagnosis; no model inference or GT-time query choice.

Pickle inputs are trusted project caches. --blind-only and --self-test need no
torch. Main audit supports --input-meta-dir portable JSON exports, otherwise
loads original torch-containing inputs on the configured server environment.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import argparse
import csv
import hashlib
import json
import math
import pickle
import runpy
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(2 ** 20), b''):
            h.update(part)
    return h.hexdigest()


def fraction(n, d):
    return {'numerator': int(n), 'denominator': int(d),
            'fraction': float(n / d) if d else None}


def is_true(v):
    if str(v).lower() in ('true', '1'):
        return True
    if str(v).lower() in ('false', '0'):
        return False
    raise ValueError('Unrecognized boolean: ' + repr(v))


def array(x):
    if hasattr(x, 'detach'):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    if not rows:
        Path(path).write_text('', encoding='utf-8')
        return
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def duration_bin(d):
    return '<1' if d < 1 else '1-3' if d < 3 else '3-10' if d < 10 else '>=10'


def coexist_bin(d):
    return '<3' if d < 3 else '3-10' if d < 10 else '>=10'


def temporal_iou(pred, gt):
    if pred is None or pred[1] <= pred[0]:
        return 0.0
    inter = max(0, min(pred[1], gt[1]) - max(pred[0], gt[0]))
    return float(inter / ((pred[1] - pred[0]) + (gt[1] - gt[0]) - inter))


def probability32(logits):
    logits = np.asarray(logits, dtype=np.float32)
    if not np.isfinite(logits).all():
        raise ValueError('Nonfinite mask logits')
    with np.errstate(over='ignore'):
        return np.float32(1.) / (np.float32(1.) + np.exp(-logits))


def class_probability(logits):
    """Original classifier softmax over all 51 classes, including background."""
    logits = np.asarray(logits, dtype=np.float32)
    if not np.isfinite(logits).all():
        raise ValueError('Nonfinite classification logits')
    # Use torch's original float32 operation when torch is available. Portable
    # JSON/CPU fallback is recorded and numerically checked against any supplied
    # reference at the candidate-selection boundary.
    try:
        import torch
    except (ImportError, OSError):
        a = np.exp(logits - logits.max(axis=-1, keepdims=True))
        return a / a.sum(axis=-1, keepdims=True), 'numpy_float32_softmax'
    return torch.from_numpy(logits.copy()).softmax(-1).numpy(), 'torch_float32_softmax'


def decode(logits, frames):
    positive = probability32(logits) > np.float32(.5)
    ids = np.flatnonzero(positive)
    if not len(ids):
        return None, 0, 0
    # Frozen evaluate_masks.make_candidate: right edge +1, NOT stride4.
    interval = (int(frames[ids[0]]), int(frames[ids[-1]]) + 1)
    return interval, int(len(ids)), int(1 + np.sum(np.diff(ids) > 1))


def get_rows(path, selection):
    rows = [r for r in read_csv(path) if r['video_name'] in selection]
    keys = [(r['video_name'], int(r['gt_index'])) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate video/GT keys')
    for r in rows:
        fps, d = float(r['fps']), float(r['relation_sec'])
        joint = float(r['joint_local_continuous_sec'])
        if not all(math.isfinite(v) and v > 0 for v in (fps, d, joint)):
            raise ValueError('Invalid FPS or duration')
        if not math.isclose((int(r['end_fid']) - int(r['begin_fid'])) / fps, d,
                            rel_tol=1e-8, abs_tol=1e-8):
            raise ValueError('Metadata frames/FPS/duration mismatch')
        r['duration_bin'] = duration_bin(d)
        r['coexist_bin'] = coexist_bin(float(r['joint_local_continuous_sec']))
    if set(r['video_name'] for r in rows) != set(selection):
        raise ValueError('Missing selected video metadata')
    return rows


def make_blind_manifest(rows, limit, seed):
    """Predetermined greedy matching, no model outcomes/scores accessed."""
    rng = np.random.default_rng(seed)
    eligible = [r for r in rows if is_true(r['F']) and float(r['relation_sec']) < 3]
    targets = [r for r in eligible if float(r['joint_local_continuous_sec']) >= 10]
    controls = [r for r in eligible if float(r['joint_local_continuous_sec']) < 10]
    target_order = rng.permutation(len(targets))
    control_tiebreak = rng.random(len(controls))
    used, matched = set(), []
    for ti in target_order:
        t = targets[int(ti)]
        options = [(abs(float(c['relation_sec']) - float(t['relation_sec'])),
                    float(control_tiebreak[i]), i)
                   for i, c in enumerate(controls) if i not in used
                   and c['predicate'] == t['predicate']
                   and abs(float(c['relation_sec']) - float(t['relation_sec'])) <= .25 + 1e-12]
        if not options:
            continue
        _, _, ci = min(options)
        used.add(ci)
        matched.append((t, controls[ci]))
        if len(matched) >= limit:
            break
    out = []
    for mi, pair in enumerate(matched):
        for role, r in zip(('target', 'control'), pair):
            # Mask group/outcome/class-probability deliberately absent.
            out.append({'match_id': mi, 'role': role, 'video_name': r['video_name'],
                        'gt_index': int(r['gt_index']), 'subject_tid': int(r['subject_tid']),
                        'object_tid': int(r['object_tid']), 'predicate': r['predicate'],
                        'subject_class': r['subject_class'], 'object_class': r['object_class'],
                        'begin_fid': int(r['begin_fid']), 'end_fid': int(r['end_fid']),
                        'fps': float(r['fps']), 'relation_sec': float(r['relation_sec']),
                        'joint_local_continuous_sec': float(r['joint_local_continuous_sec']),
                        'native_input_feasible': True, 'visual_clarity': '', 'occlusion': '',
                        'start_boundary_clear': '', 'end_boundary_clear': '',
                        'annotation_credible': '', 'reviewer_note': ''})
    summary = {'seed': seed, 'requested_pairs': limit, 'matched_pairs': len(matched),
               'eligible_targets': len(targets), 'eligible_controls': len(controls),
               'predicate_exact': True, 'max_duration_difference_seconds': .25,
               'prediction_outcome_used': False, 'labels_are_unfilled': True,
               'sampler': 'seeded target permutation; nearest eligible control; no reuse; greedy',
               'target': 'GT27, native F, relation<3s, GT joint continuous support>=10s',
               'control': 'GT27, native F, relation<3s, GT joint continuous support<10s'}
    return out, summary


def endpoint_simulation(duration, delta):
    result = {}
    for s, e, name in ((-1, -1, 'translate_early'), (-1, 1, 'expand'),
                       (1, -1, 'contract'), (1, 1, 'translate_late')):
        result['metric_only_' + name + '_tiou'] = temporal_iou((s * delta, duration + e * delta), (0., duration))
    return result


def load_candidates(video, args, hashes):
    pair_path = args.pairs_dir / (video + '.json')
    raw_path = args.raw_dir / (video + '.npz')
    input_path = (args.input_meta_dir / (video + '.json') if args.input_meta_dir
                  else args.input_dir / (video + '.pkl'))
    for p in (pair_path, raw_path, input_path):
        hashes[str(p.resolve())] = sha(p)
    if args.input_meta_dir:
        inp = json.loads(input_path.read_text())
    else:
        with input_path.open('rb') as f:
            inp = pickle.load(f)
    tids = [int(x) for x in inp['gt_tids']]
    durations = array(inp['traj_durations']).astype(int)
    sids, oids = array(inp['sids']).astype(int), array(inp['oids']).astype(int)
    if durations.shape != (len(tids), 2):
        raise ValueError('GT identity/input duration dimensions disagree')
    pairs = json.loads(pair_path.read_text())
    if len(sids) != len(pairs) or len(oids) != len(pairs):
        raise ValueError('Input pair count mismatch')
    if sorted(int(p['pair_idx']) for p in pairs) != list(range(len(pairs))):
        raise ValueError('Raw pair indexes are not unique contiguous indexes')
    by_identity, n = defaultdict(list), 0
    with np.load(raw_path, allow_pickle=False) as raw:
        if raw['pred_logits'].shape != (len(pairs), 9, 51):
            raise ValueError('Expected original 9-query/51-class outputs')
        probs, softmax_impl = class_probability(raw['pred_logits'])
        for p in pairs:
            i, s, o = (int(p[x]) for x in ('pair_idx', 'sid', 'oid'))
            if (s, o) != (sids[i], oids[i]):
                raise ValueError('Raw/input entity index mismatch')
            start, end, offset = (int(p[x]) for x in ('pair_start', 'pair_end', 'offset'))
            if not (max(durations[s, 0], durations[o, 0]) <= start < end <=
                    min(durations[s, 1], durations[o, 1]) and 0 <= offset < 4):
                raise ValueError('Invalid pair support')
            frames = np.arange(start + offset, end, 4)
            masks = raw['mask_' + str(i)]
            if masks.shape != (9, len(frames)):
                raise ValueError('Valid sampled mask length mismatch')
            for q in range(9):
                interval, positive, runs = decode(masks[q], frames)
                by_identity[tids[s], tids[o]].append({
                    'pair_idx': i, 'query': q, 'class_probs': probs[i, q],
                    'interval': interval, 'positive_points': positive, 'positive_runs': runs,
                    'frames': frames, 'pair_start': start, 'pair_end': end})
                n += 1
    return by_identity, n, softmax_impl


def summarize(rows):
    n = len(rows)
    output = [r for r in rows if not r['no_response'] and not r['no_pair']]
    result = {'n_gt': n, 'n_videos': len({r['video_name'] for r in rows}),
              'no_pair': fraction(sum(r['no_pair'] for r in rows), n),
              'no_response': fraction(sum(r['no_response'] for r in rows), n),
              'has_raw_interval': fraction(len(output), n),
              'min5_rejected': fraction(sum(r['min5_rejected'] for r in rows), n),
              'chosen_pair_disjoint_from_gt': fraction(sum(r['chosen_pair_disjoint_from_gt'] for r in rows), n),
              'chosen_pair_covers_gt': fraction(sum(r['chosen_pair_covers_gt'] for r in rows), n),
              'main_tiou50_raw': fraction(sum(r['main_tiou'] >= .5 for r in rows), n),
              'main_tiou50_min5': fraction(sum(r['main_tiou_min5'] >= .5 for r in rows), n),
              'oracle_tiou50_min5': fraction(sum(r['oracle_tiou_min5'] >= .5 for r in rows), n)}
    for flag in ('overlong', 'underlong', 'start_early', 'start_late', 'end_early', 'end_late'):
        result[flag + '_among_output'] = fraction(sum(r[flag] for r in output), len(output))
    for metric in ('main_tiou', 'main_tiou_min5', 'oracle_tiou_min5', 'gt_grid_points'):
        values = [float(r[metric]) for r in rows]
        result[metric + '_all_gt'] = {'mean': float(np.mean(values)) if n else None,
                                     'median': float(np.median(values)) if n else None, 'n': n}
    for metric in ('start_error_sec', 'end_error_sec', 'start_abs_error_sec', 'end_abs_error_sec',
                   'start_abs_normalized', 'end_abs_normalized', 'length_ratio'):
        values = [float(r[metric]) for r in output]
        result[metric + '_among_output'] = {'mean': float(np.mean(values)) if values else None,
            'median': float(np.median(values)) if values else None, 'n_output': len(values), 'n_all': n}
    for name in ('translate_early', 'expand', 'contract', 'translate_late'):
        result['metric_only_' + name + '_tiou50'] = fraction(
            sum(r['metric_only_' + name + '_tiou'] >= .5 for r in rows), n)
    return result


def main(args):
    args.out_dir.mkdir(parents=True, exist_ok=True)
    selection = json.loads(args.selection.read_text())
    if not isinstance(selection, list) or len(selection) != len(set(selection)):
        raise ValueError('Expected unique video list')
    rows = get_rows(args.per_gt, selection)
    if len(rows) != args.expected_gt or len(selection) != args.expected_videos:
        raise ValueError('Frozen GT/video denominator mismatch')
    manifest = {str(p.resolve()): sha(p) for p in (Path(__file__), args.per_gt, args.selection)}
    blind, blind_summary = make_blind_manifest(rows, args.blind_pairs, args.seed)
    write_csv(args.out_dir / 'blind_sample.csv', blind)
    (args.out_dir / 'blind_sample_summary.json').write_text(json.dumps(blind_summary, indent=2))
    if args.blind_only:
        (args.out_dir / 'source_manifest.json').write_text(json.dumps(manifest, indent=2))
        print(json.dumps(blind_summary))
        return
    if not args.category_py or not args.raw_dir or not (args.input_dir or args.input_meta_dir):
        raise ValueError('Main audit requires category, raw, and input paths')
    manifest[str(args.category_py.resolve())] = sha(args.category_py)
    cats = runpy.run_path(str(args.category_py))['vidor_pred_categories']
    if len(cats) != 51 or cats[0] != '__background__':
        raise ValueError('Unexpected predicate vocabulary')
    pids = {p: i for i, p in enumerate(cats)}
    if args.evaluator:
        manifest[str(args.evaluator.resolve())] = sha(args.evaluator)
        ev = runpy.run_path(str(args.evaluator))
        for mask in (np.array([-10., 0., 10.]), np.array([0., 0., 0.]), np.array([1., -1., 1.])):
            frame = np.array([0, 4, 8])
            ours = decode(mask, frame)[0]
            positive = ev['sigmoid_positive'](mask)
            ids = np.flatnonzero(positive)
            expected = (int(frame[ids[0]]), int(frame[ids[-1]]) + 1) if len(ids) else None
            if ours != expected:
                raise ValueError('Frozen decoder does not close')
    by_video = defaultdict(list)
    for r in rows:
        by_video[r['video_name']].append(r)
    out, softmax_implementations = [], set()
    for video in selection:
        candidates, n_queries, impl = load_candidates(video, args, manifest)
        softmax_implementations.add(impl)
        # Cache choices only by identity/predicate, never GT event time.
        chosen = {}
        for r in by_video[video]:
            s, o, pred = int(r['subject_tid']), int(r['object_tid']), r['predicate']
            pid = pids[pred]
            key = (s, o, pred)
            cs = candidates.get((s, o), [])
            if key not in chosen:
                chosen[key] = min(cs, key=lambda c: (-float(c['class_probs'][pid]), c['pair_idx'], c['query'])) if cs else None
            c = chosen[key]
            fps, d = float(r['fps']), float(r['relation_sec'])
            gt = (int(r['begin_fid']), int(r['end_fid']))
            interval = c['interval'] if c else None
            min5 = interval is not None and interval[1] - interval[0] >= 5
            tiou = temporal_iou(interval, gt)
            oracle = max((temporal_iou(x['interval'], gt) for x in cs
                          if x['interval'] is not None and x['interval'][1] - x['interval'][0] >= 5), default=0.)
            all_grid = sorted({int(f) for x in cs for f in x['frames']})
            outrow = {'video_name': video, 'gt_index': int(r['gt_index']), 'subject_tid': s,
                      'object_tid': o, 'predicate': pred, 'begin_fid': gt[0], 'end_fid': gt[1],
                      'fps': fps, 'relation_sec': d, 'duration_bin': r['duration_bin'],
                      'duration_quarter_sec_bin': int(math.floor(d / .25)),
                      'joint_local_continuous_sec': float(r['joint_local_continuous_sec']),
                      'coexist_bin': r['coexist_bin'], 'native_F': is_true(r['F']),
                      'pair_idx': c['pair_idx'] if c else '', 'query': c['query'] if c else '',
                      'target_class_probability': float(c['class_probs'][pid]) if c else '',
                      'n_identity_queries': len(cs), 'no_pair': c is None,
                      'n_identity_pair_segments': len({x['pair_idx'] for x in cs}),
                      'chosen_pair_begin_fid': c['pair_start'] if c else '',
                      'chosen_pair_end_fid': c['pair_end'] if c else '',
                      'chosen_pair_covers_gt': c is not None and c['pair_start'] <= gt[0] and c['pair_end'] >= gt[1],
                      'chosen_pair_disjoint_from_gt': c is not None and (c['pair_end'] <= gt[0] or c['pair_start'] >= gt[1]),
                      'no_response': c is not None and interval is None,
                      'min5_rejected': interval is not None and not min5,
                      'pred_begin_fid': interval[0] if interval else '',
                      'pred_end_fid': interval[1] if interval else '',
                      'main_tiou': tiou, 'main_tiou_min5': tiou if min5 else 0.,
                      'oracle_tiou_min5': oracle,
                      'gt_grid_points': sum(gt[0] <= f < gt[1] for f in all_grid),
                      'chosen_query_gt_grid_points': int(np.sum((c['frames'] >= gt[0]) & (c['frames'] < gt[1]))) if c else 0,
                      'positive_points': c['positive_points'] if c else 0,
                      'positive_runs': c['positive_runs'] if c else 0,
                      'start_error_sec': '', 'end_error_sec': '', 'start_abs_error_sec': '',
                      'end_abs_error_sec': '', 'start_abs_normalized': '', 'end_abs_normalized': '',
                      'length_ratio': '', 'overlong': False, 'underlong': False,
                      'start_early': False, 'start_late': False, 'end_early': False, 'end_late': False}
            if interval:
                ds, de = (interval[0] - gt[0]) / fps, (interval[1] - gt[1]) / fps
                ratio = (interval[1] - interval[0]) / (gt[1] - gt[0])
                outrow.update(start_error_sec=ds, end_error_sec=de,
                              start_abs_error_sec=abs(ds), end_abs_error_sec=abs(de),
                              start_abs_normalized=abs(ds) / d, end_abs_normalized=abs(de) / d,
                              length_ratio=ratio, overlong=ratio > 1, underlong=ratio < 1,
                              start_early=ds < 0, start_late=ds > 0, end_early=de < 0, end_late=de > 0)
            outrow.update(endpoint_simulation(d, args.metric_delta))
            out.append(outrow)
        print(json.dumps({'video': video, 'GT': len(by_video[video]), 'queries': n_queries}), flush=True)
    reuse = Counter((r['video_name'], r['subject_tid'], r['object_tid'], r['predicate']) for r in out)
    for r in out:
        r['identity_predicate_GT_count'] = reuse[r['video_name'], r['subject_tid'], r['object_tid'], r['predicate']]
    overall = summarize(out)
    if args.expected_oracle_hits >= 0 and overall['oracle_tiou50_min5']['numerator'] != args.expected_oracle_hits:
        raise ValueError('Historical any-query hull coverage does not close: ' + str(overall))
    write_csv(args.out_dir / 'per_gt_boundary.csv', out)
    result = {'complete': True, 'diagnostic_not_formal_recall': True,
              'query_selection': 'max original target-predicate softmax probability over all identity-compatible queries; no GT time',
              'softmax_implementations': sorted(softmax_implementations),
              'oracle': 'best tIoU across all identity queries, GT-assisted, min5',
              'decoder': 'float32 sigmoid>0.5; [first positive sampled frame,last positive sampled frame+1); stride4',
              'metric_delta_seconds': args.metric_delta,
              'metric_simulation_not_model_experiment': True,
              'reused_identity_predicate_groups': sum(v > 1 for v in reuse.values()),
              'GT_in_reused_groups': sum(v for v in reuse.values() if v > 1),
              'multi_segment_note': 'GT identity can span several continuous input segments; main class-only selection uses no GT time and may select a different segment; this is counted separately, not interpreted as boundary-only failure.',
              'overall': overall, 'groups': {}}
    for fields in (('duration_bin',), ('predicate',), ('coexist_bin',), ('native_F',),
                   ('predicate', 'duration_bin'), ('duration_bin', 'coexist_bin'),
                   ('predicate', 'duration_bin', 'coexist_bin'), ('video_name',)):
        groups = defaultdict(list)
        for r in out:
            groups[tuple(str(r[f]) for f in fields)].append(r)
        result['groups']['__'.join(fields)] = {'|'.join(k): summarize(v) for k, v in sorted(groups.items())}
    (args.out_dir / 'stratified_summary.json').write_text(json.dumps(result, indent=2))
    (args.out_dir / 'source_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({'complete': True, 'overall': overall, 'out_dir': str(args.out_dir)}, indent=2))


def self_test():
    assert decode(np.array([1., -1., 1.]), np.array([2, 6, 10]))[0] == (2, 11)
    assert decode(np.zeros(3), np.array([0, 4, 8]))[0] is None
    assert decode(np.array([1., -1.]), np.array([0, 4]))[0] == (0, 1)
    assert temporal_iou((0, 10), (2, 6)) == .4
    assert temporal_iou(None, (0, 10)) == 0
    assert math.isclose(endpoint_simulation(1, .2)['metric_only_expand_tiou'], 1 / 1.4)
    assert endpoint_simulation(.3, .2)['metric_only_contract_tiou'] == 0
    # A long context with high absolute wrong mask must not be selected by GT IoU.
    cs = [{'p': .8, 'interval': (100, 200)}, {'p': .7, 'interval': (0, 10)}]
    c = max(cs, key=lambda x: x['p'])
    assert temporal_iou(c['interval'], (0, 10)) == 0
    print('SELF_TEST_PASS')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('raw-dir', 'pairs-dir', 'input-dir', 'input-meta-dir', 'per-gt', 'selection',
                 'category-py', 'evaluator', 'out-dir'):
        p.add_argument('--' + name, type=Path)
    p.add_argument('--expected-gt', type=int, default=2573)
    p.add_argument('--expected-videos', type=int, default=27)
    p.add_argument('--expected-oracle-hits', type=int, default=1731)
    p.add_argument('--blind-pairs', type=int, default=40)
    p.add_argument('--seed', type=int, default=77)
    p.add_argument('--metric-delta', type=float, default=.2)
    p.add_argument('--blind-only', action='store_true')
    p.add_argument('--self-test', action='store_true')
    a = p.parse_args()
    if a.self_test:
        self_test()
    else:
        if any(getattr(a, x) is None for x in ('per_gt', 'selection', 'out_dir')):
            p.error('--per-gt --selection --out-dir required')
        if a.pairs_dir is None:
            a.pairs_dir = a.raw_dir
        main(a)
