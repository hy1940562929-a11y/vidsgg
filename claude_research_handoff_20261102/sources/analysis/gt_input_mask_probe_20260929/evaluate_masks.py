#!/usr/bin/env python3
"""CPU, GT-assisted raw-mask availability audit; never an achieved detection Recall.

Inputs are trusted, frozen local run caches. No inference, training, SSH, topK,
predicate scoring, external evaluator package, or ijson is used. --self-test runs
small coordinate/matching fixtures without reading run caches or writing files.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')

import argparse
import csv
import hashlib
import json
import pickle
import runpy
import time
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np


DEFAULT_ROOT = '/root/rebinding_project/VrdONE'
DEFAULT_META = '/root/autodl-tmp/VrdONE-data/big_vs_vrdone_20260928_163048/instance_comparison.csv'
THRESHOLDS = {'viou50': ('viou', .5), 'tiou30': ('tiou', .3),
              'tiou50': ('tiou', .5), 'tiou70': ('tiou', .7), 'binary50': ('binary_iou', .5)}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 ** 2), b''):
            digest.update(block)
    return digest.hexdigest()


def array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def prep(relation):
    """Copied audit_saved.py volume convention: inclusive XY box coordinates."""
    boxes = np.asarray([relation['sub_traj'], relation['obj_traj']], dtype=np.float64)
    if boxes.shape != (2, relation['duration'][1] - relation['duration'][0], 4):
        raise ValueError('Trajectory length does not equal half-open interval length')
    area = (boxes[:, :, 2] - boxes[:, :, 0] + 1) * (boxes[:, :, 3] - boxes[:, :, 1] + 1)
    if not np.isfinite(boxes).all() or (area <= 0).any():
        raise ValueError('Invalid trajectory box or area')
    return boxes, area.sum(axis=1), area


def overlap(prediction, truth, prepared_prediction, prepared_truth):
    """Official full-duration min(subject vIoU, object vIoU), not cropped IoU."""
    start = max(prediction['duration'][0], truth['duration'][0])
    end = min(prediction['duration'][1], truth['duration'][1])
    if start >= end:
        return 0.
    pb, pvolume, _ = prepared_prediction
    gb, gvolume, _ = prepared_truth
    x = pb[:, start - prediction['duration'][0]:end - prediction['duration'][0]]
    y = gb[:, start - truth['duration'][0]:end - truth['duration'][0]]
    wh = np.maximum(0, np.minimum(x[:, :, 2:], y[:, :, 2:]) - np.maximum(x[:, :, :2], y[:, :, :2]) + 1)
    intersection = (wh[:, :, 0] * wh[:, :, 1]).sum(axis=1)
    return float(np.min(intersection / (pvolume + gvolume - intersection)))


def temporal_iou(predicted, truth):
    inter = max(0, min(predicted[1], truth[1]) - max(predicted[0], truth[0]))
    union = predicted[1] - predicted[0] + truth[1] - truth[0] - inter
    return inter / union if union else 0.


def forward_hold_spans(frames, positive, pair_end):
    """Union of [sample_frame,sample_frame+4), clipped only to pair_end.

    Evaluation later intersects this global support with GT, but the denominator
    retains every predicted frame, including all frames outside the GT interval.
    """
    spans = []
    for frame in frames[positive]:
        start, end = int(frame), min(int(frame) + 4, int(pair_end))
        if end <= start:
            continue
        if spans and start <= spans[-1][1]:
            spans[-1][1] = max(end, spans[-1][1])
        else:
            spans.append([start, end])
    return spans


def binary_frame_iou(spans, truth):
    size = sum(end - start for start, end in spans)
    inter = sum(max(0, min(end, truth[1]) - max(start, truth[0])) for start, end in spans)
    union = size + truth[1] - truth[0] - inter
    return inter / union if union else 0.


def sigmoid_positive(logits):
    """Replicate a float32 sigmoid>0.5, including its rounding around zero."""
    logits = np.asarray(logits, dtype=np.float32)
    if not np.isfinite(logits).all():
        raise ValueError('Nonfinite raw mask logit')
    with np.errstate(over='ignore'):
        return np.float32(1.) / (np.float32(1.) + np.exp(-logits)) > np.float32(.5)


def relation_key(relation):
    return (int(relation['subject_tid']), int(relation['object_tid']), relation['predicate'],
            int(relation['begin_fid']), int(relation['end_fid']))


def load_gt(annotation_path, metadata_rows):
    annotation = json.loads(annotation_path.read_text())
    categories = {int(item['tid']): item['category'] for item in annotation['subject/objects']}
    tracks = defaultdict(dict)
    for frame, items in enumerate(annotation['trajectories']):
        for item in items:
            box = item['bbox']
            tid = int(item['tid'])
            if frame in tracks[tid]:
                raise ValueError('Duplicate GT tid within an annotation frame')
            tracks[tid][frame] = [box['xmin'], box['ymin'], box['xmax'], box['ymax']]
    occurrences = defaultdict(deque)
    for annotation_index, relation in enumerate(annotation['relation_instances']):
        occurrences[relation_key(relation)].append(annotation_index)
    if Counter(relation_key(r) for r in annotation['relation_instances']) != Counter(relation_key(r) for r in metadata_rows):
        raise ValueError('Annotation and metadata relation multisets differ; refusing to alter the GT denominator')
    truths, seen = [], Counter()
    for meta in sorted(metadata_rows, key=lambda row: int(row['gt_index'])):
        stid, otid, predicate, begin, end = key = relation_key(meta)
        if not 0 <= begin < end <= int(annotation['frame_count']):
            raise ValueError('GT duration is not a valid 0-based half-open interval')
        if [meta['subject_class'], meta['object_class']] != [categories[stid], categories[otid]]:
            raise ValueError('Metadata and annotation category mismatch')
        if meta['group_at100'] not in {'A', 'B', 'C', 'D'}:
            raise ValueError('Missing original A/B/C/D group')
        try:
            subject = [tracks[stid][frame] for frame in range(begin, end)]
            obj = [tracks[otid][frame] for frame in range(begin, end)]
        except KeyError as error:
            raise ValueError('GT relation crosses an unannotated track frame') from error
        truth = {'duration': [begin, end], 'sub_traj': subject, 'obj_traj': obj,
                 'stid': stid, 'otid': otid, 'triplet': [categories[stid], predicate, categories[otid]],
                 'gt_index': int(meta['gt_index']), 'annotation_index': occurrences[key].popleft(),
                 'tuple_occurrence': seen[key], 'meta': meta,
                 'duration_sec': (end - begin) / float(annotation['fps'])}
        seen[key] += 1
        truth['prepared'] = prep(truth)
        truths.append(truth)
    if len({g['gt_index'] for g in truths}) != len(truths):
        raise ValueError('Duplicate metadata gt_index within a video')
    return annotation, truths


def make_candidate(pair, positive, boxes, durations, candidate_id, query):
    positions = np.flatnonzero(positive)
    if not len(positions):
        return None
    frames = pair['frames']
    begin, end = int(frames[positions[0]]), int(frames[positions[-1]]) + 1
    sid, oid = pair['sid'], pair['oid']
    prediction = {'duration': [begin, end],
        'sub_traj': boxes[sid][begin - durations[sid, 0]:end - durations[sid, 0]],
        'obj_traj': boxes[oid][begin - durations[oid, 0]:end - durations[oid, 0]],
        'id': candidate_id, 'pair_idx': pair['pair_idx'], 'query': query, 'sid': sid, 'oid': oid,
        'min5': end - begin >= 5, 'spans': forward_hold_spans(frames, positive, pair['pair_end']),
        'positive_points': len(positions), 'positive_runs': 1 + int((np.diff(positions) > 1).sum())}
    prediction['prepared'] = prep(prediction)
    return prediction


def load_arm(run, condition, full, annotation, category_names, hashes):
    paths = {'input': run / condition / 'inputs' / (full + '.pkl'),
             'pairs': run / condition / 'raw' / (full + '.json'),
             'raw': run / condition / 'raw' / (full + '.npz')}
    present = {name: path.is_file() for name, path in paths.items()}
    if not all(present.values()):
        raise FileNotFoundError('Missing cached arm artifacts are not an explicitly empty input: ' + str(paths))
    for path in paths.values():
        hashes[str(path.resolve())] = sha(path)
    with paths['input'].open('rb') as stream:
        inputs = pickle.load(stream)
    pair_rows = json.loads(paths['pairs'].read_text())
    if not pair_rows:
        if len(inputs.get('sids', [])) or len(inputs.get('oids', [])):
            raise ValueError('Empty raw pair list disagrees with nonempty cached input')
        with np.load(paths['raw'], allow_pickle=False) as raw:
            if raw['pred_logits'].shape != (0, 9, 51):
                raise ValueError('Empty input requires explicit [0,9,51] raw logits')
        return {'available': True, 'reason': 'empty_input_pairs', 'pairs': [],
                'candidates': [], 'by_key': {}, 'boxes': [], 'durations': []}
    durations = array(inputs['traj_durations']).astype(np.int64)
    boxes = [array(value).astype(np.float64, copy=True) for value in inputs['bboxes_list']]
    categories = [category_names[int(category)] for category in array(inputs['cat_ids'])]
    if durations.shape != (len(boxes), 2) or len(categories) != len(boxes):
        raise ValueError('Input trajectory/category dimensions disagree')
    tids = [int(tid) for tid in inputs.get('gt_tids', [])]
    if condition == 'gt' and len(tids) != len(boxes):
        raise ValueError('GT arm requires an explicit GT tid for every track segment')
    for index, box in enumerate(boxes):
        if box.shape != (durations[index, 1] - durations[index, 0], 4):
            raise ValueError('Input track length/duration mismatch')
        if not 0 <= durations[index, 0] < durations[index, 1] <= annotation['frame_count']:
            raise ValueError('Input track is outside the annotated video')
        # This is the original native VidOR validation box convention.
        if condition == 'native':
            box[:, 0] = np.maximum(0, box[:, 0]); box[:, 1] = np.maximum(0, box[:, 1])
            box[:, 2] = np.minimum(annotation['width'] - 1, box[:, 2])
            box[:, 3] = np.minimum(annotation['height'] - 1, box[:, 3])
    input_sids, input_oids = array(inputs['sids']).astype(int), array(inputs['oids']).astype(int)
    if len(input_sids) != len(pair_rows) or len(input_oids) != len(pair_rows):
        raise ValueError('Input and raw pair counts disagree')
    candidates, pairs, by_key = [], [], defaultdict(list)
    seen_ids = set()
    with np.load(paths['raw'], allow_pickle=False) as raw:
        if raw['pred_logits'].shape != (len(pair_rows), 9, 51):
            raise ValueError('Expected one frozen [9,51] class output per cached pair; scores are never used')
        for row in pair_rows:
            pair = {key: int(row[key]) for key in ('pair_idx', 'sid', 'oid', 'pair_start', 'pair_end', 'offset')}
            index, sid, oid = pair['pair_idx'], pair['sid'], pair['oid']
            if index in seen_ids or not 0 <= index < len(pair_rows):
                raise ValueError('Raw pair IDs must be unique contiguous input indexes')
            seen_ids.add(index)
            if index >= len(input_sids) or (sid, oid) != (input_sids[index], input_oids[index]):
                raise ValueError('Raw pair record and input sid/oid identities disagree')
            if not (0 <= sid < len(boxes) and 0 <= oid < len(boxes)):
                raise ValueError('Raw track index out of bounds')
            if not (max(durations[sid, 0], durations[oid, 0]) <= pair['pair_start'] < pair['pair_end']
                    <= min(durations[sid, 1], durations[oid, 1]) and 0 <= pair['offset'] < 4):
                raise ValueError('Invalid common pair interval or sample offset')
            frames = np.arange(pair['pair_start'] + pair['offset'], pair['pair_end'], 4, dtype=np.int64)
            masks = raw['mask_' + str(index)]
            if masks.shape != (9, len(frames)):
                raise ValueError('Raw masks must contain exactly the unpadded valid sampled positions')
            pair['frames'] = frames
            pair['key'] = (tids[sid], tids[oid]) if condition == 'gt' else (categories[sid], categories[oid])
            pair['discarded_padding_positions'] = masks.shape[1] - len(frames)
            pairs.append(pair)
            for query, logits in enumerate(masks[:, :len(frames)]):
                candidate = make_candidate(pair, sigmoid_positive(logits), boxes, durations, index * 9 + query, query)
                if candidate is not None:
                    candidates.append(candidate)
                    by_key[pair['key']].append(candidate)
    return {'available': True, 'reason': 'loaded' if pairs else 'empty_input_pairs',
            'pairs': pairs, 'candidates': candidates, 'by_key': by_key, 'boxes': boxes, 'durations': durations}


def score_candidates(candidates, truth):
    return [(candidate, {'viou': overlap(candidate, truth, candidate['prepared'], truth['prepared']),
                          'tiou': temporal_iou(candidate['duration'], truth['duration']),
                          'binary_iou': binary_frame_iou(candidate['spans'], truth['duration'])})
            for candidate in candidates]


def witness(candidate, scores):
    return {'candidate_id': candidate['id'], 'pair_idx': candidate['pair_idx'], 'query': candidate['query'],
            'sid': candidate['sid'], 'oid': candidate['oid'], 'duration': candidate['duration'],
            'min5': candidate['min5'], 'positive_points': candidate['positive_points'],
            'positive_runs': candidate['positive_runs'], 'forward_hold_spans': candidate['spans'], **scores}


def availability(scored, metric, threshold, min5):
    selected = [(candidate, values) for candidate, values in scored if candidate['min5'] or not min5]
    if not selected:
        return False, 0., None
    candidate, values = max(selected, key=lambda item: (item[1][metric], -item[0]['id']))
    return bool(values[metric] >= threshold), float(values[metric]), witness(candidate, values)


def grid_references(arm, truth):
    """Perfect sampled labels are a reference, not the arbitrary-mask upper bound."""
    perfect, best = [], {'min5': (0., None), 'unfiltered': (0., None)}
    for pair in arm['pairs']:
        if pair['key'] != (truth['stid'], truth['otid']):
            continue
        frames = pair['frames']
        labels = (frames >= truth['duration'][0]) & (frames < truth['duration'][1])
        candidate = make_candidate(pair, labels, arm['boxes'], arm['durations'], pair['pair_idx'] * 9, -1)
        if candidate is not None:
            perfect.append(candidate)
        if not len(frames):
            continue
        starts, ends = frames[:, None], frames[None, :] + 1
        lengths = ends - starts
        intersection = np.maximum(0, np.minimum(ends, truth['duration'][1]) - np.maximum(starts, truth['duration'][0]))
        union = lengths + truth['duration'][1] - truth['duration'][0] - intersection
        values = intersection / np.maximum(union, 1)
        for name, minimum in (('min5', 5), ('unfiltered', 1)):
            legal = lengths >= minimum
            if legal.any():
                masked = np.where(legal, values, -1.)
                si, ei = np.unravel_index(masked.argmax(), masked.shape)
                value = float(masked[si, ei])
                if best[name][1] is None or value > best[name][0]:
                    best[name] = (value, {'pair_idx': pair['pair_idx'], 'duration': [int(frames[si]), int(frames[ei]) + 1]})
    return score_candidates(perfect, truth), best


def maximum_matching(adjacency):
    """Deterministic DFS maximum cardinality event-to-query-mask matching."""
    candidate_to_event = {}
    def augment(event, seen):
        for candidate in adjacency[event]:
            if candidate in seen:
                continue
            seen.add(candidate)
            previous = candidate_to_event.get(candidate)
            if previous is None or augment(previous, seen):
                candidate_to_event[candidate] = event
                return True
        return False
    for event in sorted(adjacency):
        augment(event, set())
    return {event: candidate for candidate, event in candidate_to_event.items()}


def fraction(numerator, denominator):
    return {'numerator': int(numerator), 'denominator': int(denominator),
            'fraction': numerator / denominator if denominator else None}


def summarize(rows):
    n = len(rows)
    flags = sorted({key for row in rows for key, value in row.items() if isinstance(value, bool)})
    result = {'N_GT': n, 'availability': {key: fraction(sum(row.get(key, False) for row in rows), n) for key in flags}}
    for suffix in ('min5', 'unfiltered'):
        old, new = 'native_viou50_' + suffix, 'gt_viou50_' + suffix
        gained = sum(not row[old] and row[new] for row in rows)
        lost = sum(row[old] and not row[new] for row in rows)
        result['transition_' + suffix] = {'new': fraction(gained, n), 'lost': fraction(lost, n),
            'net': fraction(gained - lost, n), 'both': fraction(sum(row[old] and row[new] for row in rows), n),
            'neither': fraction(sum(not row[old] and not row[new] for row in rows), n)}
    return result


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fields or ['empty'])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, allow_nan=False) if isinstance(value, (dict, list))
                             else int(value) if isinstance(value, bool) else value for key, value in row.items()})


def self_test():
    assert forward_hold_spans(np.array([0, 4, 8]), np.array([True, False, True]), 10) == [[0, 4], [8, 10]]
    assert binary_frame_iou([[0, 4], [8, 10]], [2, 6]) == 2 / 8  # outside-GT positives remain penalized
    assert temporal_iou([0, 5], [1, 5]) == .8
    assert maximum_matching({0: [0, 1], 1: [0]}) == {1: 0, 0: 1}  # needs an augmenting path
    assert len(maximum_matching({0: [0], 1: [0]})) == 1  # one query cannot cover two events
    box = [[0., 0., 1., 1.]]
    p = {'duration': [0, 10], 'sub_traj': box * 10, 'obj_traj': box * 10}
    g = {'duration': [2, 6], 'sub_traj': box * 4, 'obj_traj': box * 4}
    assert overlap(p, g, prep(p), prep(g)) == .4  # full duration, not cropped support
    assert sigmoid_positive(np.array([-100., 0., 100.])).tolist() == [False, False, True]
    assert len({(0, 1, 10, 20), (0, 1, 10, 20)}) == 1  # predicates excluded from event key
    pair = {'frames': np.array([0, 4]), 'pair_end': 8, 'sid': 0, 'oid': 1, 'pair_idx': 0, 'key': (0, 1)}
    boxes = [np.array(box * 8), np.array(box * 8)]
    duration = np.array([[0, 8], [0, 8]])
    candidate = make_candidate(pair, np.array([False, True]), boxes, duration, 0, 0)
    assert candidate['duration'] == [4, 5] and not candidate['min5']
    truth = {'duration': [1, 5], 'sub_traj': box * 4, 'obj_traj': box * 4, 'stid': 0, 'otid': 1}
    truth['prepared'] = prep(truth)
    reference, best = grid_references({'pairs': [pair], 'boxes': boxes, 'durations': duration}, truth)
    assert not reference[0][0]['min5'] and best['min5'][0] == .8
    assert summarize([])['availability'] == {}
    return {'status': 'PASS', 'scope': 'CPU synthetic coordinate/volume/matching fixtures only; no experiment outcome'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--root', type=Path, default=Path(DEFAULT_ROOT))
    parser.add_argument('--meta', type=Path, default=Path(DEFAULT_META))
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2)); return
    if args.run is None or args.out is None:
        parser.error('--run and --out are required unless --self-test is used')
    started = time.monotonic()
    if args.out.exists():
        raise FileExistsError('Refusing to overwrite any existing evaluation directory')
    selection = json.loads((args.run / 'selection.json').read_text())
    if not isinstance(selection, list) or not all(isinstance(x, str) and len(x.split('_')) == 2 for x in selection) or len(set(selection)) != len(selection):
        raise ValueError('selection.json must contain unique group_video names')
    completion_path = args.run / 'completion.json'
    completion = json.loads(completion_path.read_text())
    if (completion.get('status') != 'RAW_INFERENCE_COMPLETE_NOT_EVALUATED'
            or completion.get('videos') != len(selection) or completion.get('training') is not False):
        raise ValueError('Complete frozen raw inference for the entire selection is required before evaluation')
    if completion.get('native_fresh') is not True:
        raise ValueError('Final comparison requires fresh native+GT outputs; legacy raw reuse failed its strict gate')
    taxonomy_path = args.root / 'dataloaders/category.py'
    category_names = runpy.run_path(str(taxonomy_path))['vidor_category_id_to_name']
    by_video = defaultdict(list)
    with args.meta.open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            by_video[row['video_name']].append(row)
    hashes = {str(path.resolve()): sha(path) for path in (Path(__file__), args.meta, args.run / 'selection.json', completion_path, taxonomy_path,
              args.run / 'effective_config.json', args.run / 'source_hashes.json')}
    args.out.mkdir(parents=True, exist_ok=False)
    all_rows, all_events, videos = [], [], []
    for full in selection:
        group, video_id = full.split('_')
        annotation_path = args.root / 'datasets/vidor/annotations/validation' / group / (video_id + '.json')
        hashes[str(annotation_path.resolve())] = sha(annotation_path)
        annotation, truths = load_gt(annotation_path, by_video.get(full, []))
        arms = {condition: load_arm(args.run, condition, full, annotation, category_names, hashes) for condition in ('native', 'gt')}
        rows, event_truths, event_scores = [], {}, {}
        for truth in truths:
            event_key = (truth['stid'], truth['otid'], *truth['duration'])
            event_truths.setdefault(event_key, []).append(truth)
            scored_native = score_candidates(arms['native']['by_key'].get((truth['triplet'][0], truth['triplet'][2]), []), truth)
            scored_gt = score_candidates(arms['gt']['by_key'].get((truth['stid'], truth['otid']), []), truth)
            event_scores.setdefault(event_key, scored_gt)
            row = {'video_id': video_id, 'video_name': full, 'gt_index': truth['gt_index'],
                'annotation_index': truth['annotation_index'], 'tuple_occurrence': truth['tuple_occurrence'],
                'subject_tid': truth['stid'], 'object_tid': truth['otid'], 'subject_class': truth['triplet'][0],
                'predicate': truth['triplet'][1], 'object_class': truth['triplet'][2],
                'begin_fid': truth['duration'][0], 'end_fid': truth['duration'][1], 'duration_sec': truth['duration_sec'],
                'group_at100': truth['meta']['group_at100'],
                'duration_bin': '<1s' if truth['duration_sec'] < 1 else '1-3s' if truth['duration_sec'] < 3 else '3-10s' if truth['duration_sec'] < 10 else '>=10s',
                'native_cache_available': arms['native']['available'], 'gt_cache_available': arms['gt']['available'],
                'native_compatible_raw_masks': len(scored_native), 'gt_correct_tid_raw_masks': len(scored_gt)}
            for condition, scored in (('native', scored_native), ('gt', scored_gt)):
                for name, (metric, threshold) in THRESHOLDS.items():
                    if condition == 'native' and metric != 'viou':
                        continue
                    for suffix, min5 in (('min5', True), ('unfiltered', False)):
                        hit, best, match = availability(scored, metric, threshold, min5)
                        key = condition + '_' + name + '_' + suffix
                        row[key], row[key + '_best'], row[key + '_witness'] = hit, best, match
            perfect, upper = grid_references(arms['gt'], truth)
            for suffix, min5 in (('min5', True), ('unfiltered', False)):
                for name, (metric, threshold) in THRESHOLDS.items():
                    hit, best, match = availability(perfect, metric, threshold, min5)
                    key = 'perfect_grid_labels_' + name + '_' + suffix
                    row[key], row[key + '_best'], row[key + '_witness'] = hit, best, match
                row['grid_hull_tiou_upper_' + suffix], row['grid_hull_tiou_upper_' + suffix + '_witness'] = upper[suffix]
                for label, threshold in (('30', .3), ('50', .5), ('70', .7)):
                    row['grid_hull_tiou' + label + '_feasible_' + suffix] = upper[suffix][0] >= threshold
                old, new = row['native_viou50_' + suffix], row['gt_viou50_' + suffix]
                row['transition_' + suffix] = 'new' if new and not old else 'lost' if old and not new else 'both' if old else 'neither'
                row['net_' + suffix] = int(new) - int(old)
            rows.append(row)
        events = []
        keys = sorted(event_truths)
        for index, key in enumerate(keys):
            members = event_truths[key]
            events.append({'video_name': full, 'event_index': index, 'subject_tid': key[0], 'object_tid': key[1],
                'begin_fid': key[2], 'end_fid': key[3], 'relation_count': len(members),
                'gt_indices': [truth['gt_index'] for truth in members],
                'predicates': sorted({truth['triplet'][1] for truth in members}),
                'contains_C': any(truth['meta']['group_at100'] == 'C' for truth in members)})
        matching_summary = {}
        for name, (metric, threshold) in THRESHOLDS.items():
            for suffix, min5 in (('min5', True), ('unfiltered', False)):
                adjacency = {index: [candidate['id'] for candidate, values in sorted(event_scores[key], key=lambda x: (-x[1][metric], x[0]['id']))
                                     if values[metric] >= threshold and (candidate['min5'] or not min5)]
                             for index, key in enumerate(keys)}
                matched = maximum_matching(adjacency)
                label = name + '_' + suffix
                matching_summary[label] = {'matched': fraction(len(matched), len(events)),
                    'independent_existence': fraction(sum(bool(x) for x in adjacency.values()), len(events))}
                for index, event in enumerate(events):
                    event[label + '_independent_exists'] = bool(adjacency[index])
                    event[label + '_matched'] = index in matched
                    event[label + '_witness'] = next((witness(c, s) for c, s in event_scores[keys[index]] if c['id'] == matched.get(index)), None)
        video = {'video_name': full, 'N_GT': len(rows), 'N_unique_time_events': len(events),
                 'relation_availability': summarize(rows), 'event_maximum_matching': matching_summary,
                 'inputs': {condition: {'available': arm['available'], 'reason': arm['reason'],
                             'pairs': len(arm['pairs']), 'nonempty_query_masks': len(arm['candidates']),
                             'discarded_padding_positions': sum(pair['discarded_padding_positions'] for pair in arm['pairs'])}
                            for condition, arm in arms.items()}}
        videos.append(video); all_rows.extend(rows); all_events.extend(events)
        print(json.dumps({'video': full, 'N_GT': len(rows), 'N_unique_events': len(events),
                          'GT_done': len(all_rows), 'elapsed_seconds': round(time.monotonic() - started, 2)}), flush=True)
    write_csv(args.out / 'per_gt.csv', all_rows)
    write_csv(args.out / 'unique_time_events.csv', all_events)
    (args.out / 'per_video.json').write_text(json.dumps(videos, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    by_duration = {name: summarize([row for row in all_rows if row['duration_bin'] == name]) for name in ('<1s', '1-3s', '3-10s', '>=10s')}
    event_summary = {name + '_' + suffix: {
        'maximum_matching': fraction(sum(row[name + '_' + suffix + '_matched'] for row in all_events), len(all_events)),
        'independent_existence': fraction(sum(row[name + '_' + suffix + '_independent_exists'] for row in all_events), len(all_events))}
        for name in THRESHOLDS for suffix in ('min5', 'unfiltered')}
    summary = {'status': 'COMPLETED', 'N_selected_videos': len(selection), 'N_GT': len(all_rows),
        'N_unique_time_events': len(all_events), 'all_GT_retained_in_denominator': True,
        'run_completion': completion,
        'effective_config': json.loads((args.run / 'effective_config.json').read_text()),
        'inference_source_hashes': json.loads((args.run / 'source_hashes.json').read_text()),
        'overall': summarize(all_rows), 'C_group': summarize([row for row in all_rows if row['group_at100'] == 'C']),
        'duration_bins': by_duration, 'unique_time_event_matching': event_summary,
        'elapsed_seconds': time.monotonic() - started,
        'definitions': {'primary': 'min(subject full-duration vIoU,object full-duration vIoU)>=.5; class-compatible native or exact GT tid pair; mask hull duration>=5',
            'hull': '[pair_start+offset+4*first_positive,pair_start+offset+4*last_positive+1)',
            'binary_support': 'Full-video frame-domain union of positive sample forward holds of4 frames, clipped at pair_end; GT-external predicted frames retained in union',
            'events': '(video,subject_tid,object_tid,begin,end), predicates deduplicated; one(pair_idx,query) can match at most one event per metric',
            'grid_perfect_labels': 'Reference obtained by perfect GT labels at sample points; NOT an upper bound over arbitrary masks',
            'grid_hull_upper': 'True maximum temporal IoU over all sampled-grid endpoint pairs in an available correct-tid input pair; not a box-vIoU upper bound'},
        'limitations': ['GT-assisted existence/matching diagnostics are not recovered official Recall and ignore predicate scores/topK.',
            'Native-versus-GT raw-mask availability combines upstream input correction, changed features, geometry and temporal support; it is not a pure temporal causal effect.',
            'GT-only temporal and binary metrics are separate from the two-arm geometry comparison.',
            'Explicit empty inputs are retained as misses; missing/partial cached arms or incomplete raw inference are errors.',
            'No matching query means unavailable within these frozen inputs and9 masks, not absence of a possible model solution.'],
        'source_sha256': hashes}
    (args.out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': summary['status'], 'N_GT': len(all_rows), 'N_unique_time_events': len(all_events),
                      'primary_transition': summary['overall']['transition_min5']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
