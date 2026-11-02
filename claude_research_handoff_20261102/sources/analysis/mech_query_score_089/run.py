#!/usr/bin/env python3
"""089 sealed CPU readout diagnostic. No model forwards or training."""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from query_math import require, select_queries, evaluate_event, summarize, decide

HERE = Path(__file__).resolve().parent
PRIMARY = 'long_coexist_short_covered'
SECONDARY = 'long_coexist_subsecond_covered'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 << 20), b''):
            h.update(b)
    return h.hexdigest()

def load(path):
    return json.loads(Path(path).read_text())

def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)

def resolve(base, name):
    p = Path(name).expanduser()
    return (p if p.is_absolute() else base/p).resolve()

def check(path, digest, receipts):
    require(path.is_file(), 'missing frozen input '+str(path))
    require(sha(path) == digest, 'SHA mismatch '+str(path))
    receipts[str(path)] = digest

def freeze_checks(manifest, protocol, software, receipts):
    for freeze in (protocol, software):
        obj = load(freeze)
        files = obj.get('files', obj.get('frozen_files'))
        require(isinstance(files, dict) and files, 'freeze files mapping missing')
        checked = set()
        for name, digest in files.items():
            path = resolve(freeze.parent, name)
            check(path, digest, receipts)
            checked.add(path)
        required = {HERE/'PROTOCOL.md', manifest} if freeze == protocol else {HERE/'run.py', HERE/'query_math.py', HERE/'test_run.py', manifest, protocol}
        require({p.resolve() for p in required} <= checked, 'freeze coverage incomplete '+str(freeze))
        receipts[str(freeze)] = sha(freeze)

def reference(base, ref, receipts):
    original = resolve(base, ref['path'])
    material = resolve(base, ref['materialized_path']) if ref.get('materialized_path') else None
    path = material if material is not None and material.is_file() else original
    check(path, ref['sha256'], receipts)
    return path

def bind(manifest_path, receipts):
    manifest = load(manifest_path)
    roles = ('members','eval_locked','coordinates','predicate_map','candidates088','summary088','rank_preflight','candidate_audits088','candidate_seal088')
    refs = {k: reference(manifest_path.parent, manifest['sources'][k], receipts) for k in roles}
    members, coords, mapping = load(refs['members']), load(refs['coordinates']), load(refs['predicate_map'])
    names = mapping['names']
    require(len(names) == len(set(names)) == 51 and names[0] == '__background__', 'predicate mapping invalid')
    require(isinstance(mapping.get('source_sha256'), str) and len(mapping['source_sha256']) == 64, 'mapping provenance missing')
    pids = {name: i for i, name in enumerate(names)}
    primary = {(str(v), int(i)) for v,i in members[PRIMARY]}
    secondary = {(str(v), int(i)) for v,i in members[SECONDARY]}
    require(len(primary) == len(members[PRIMARY]) == 400 and len(secondary) == len(members[SECONDARY]) == 163 and secondary <= primary, '400/163 membership error')
    with refs['eval_locked'].open(encoding='utf-8-sig', newline='') as f:
        raw = list(csv.DictReader(f))
    bykey = {(r['video_name'], int(r['gt_index'])): r for r in raw}
    require(len(bykey) == len(raw) == 2573 and primary <= bykey.keys(), 'locked GT duplicate/missing')
    # Class-only request construction: no time boundaries enter the selector.
    requests = {}; events = []; needed = {}
    for video, gi in sorted(primary):
        r = bykey[(video, gi)]; pi = int(r['pair_idx']); pair = f'{video}/{pi}'
        require(pair in coords and r['predicate'] in pids, 'pair/mapping missing')
        pid = pids[r['predicate']]; q = int(r['query']); key = f'{pair}/{pid}'
        require(1 <= pid <= 50 and 0 <= q < 9, 'invalid foreground/query index')
        request = dict(pair_key=pair, target_pid=pid, legacy_query=q)
        require(key not in requests or requests[key] == request, 'same pair/predicate has conflicting legacy query')
        requests[key] = request; needed[pair] = coords[pair]
        events.append(dict(video_name=video, gt_index=gi, pair_idx=pi, pair_key=pair, selection_key=key,
            predicate=r['predicate'], original_locked_query=q, subsecond163=(video,gi) in secondary))
    require(len({r['video_name'] for r in events}) == 23 and len(needed) == 161, 'fixed cohort/pair count changed')
    require(set(needed) == set(manifest['score_files']), 'score pair membership changed')
    return manifest, refs, requests, events, needed, bykey

def audit_candidates(pair, coordinate, frames):
    n = int(coordinate['N_valid'])
    require(pair['N_valid'] == n and pair['origin'] == int(coordinate['origin']), 'candidate coordinates differ')
    require(np.array_equal(pair['frames'], frames), 'candidate full frames differ')
    for arm in ('H', 'C'):
        require(len(pair[arm]) == 9, 'candidate slots differ')
        for q, interval in enumerate(pair[arm]):
            if interval is None:
                continue
            a, b = interval['first_index'], interval['last_index']
            require(isinstance(a, int) and isinstance(b, int) and 0 <= a <= b < n, 'candidate index invalid')
            require(interval['query'] == q and interval['start'] == int(frames[a]) and interval['end'] == int(frames[b])+1, 'candidate time mapping invalid')
            require(interval['frame_length'] == interval['end']-interval['start'] >= 5 and interval['sample_length'] == b-a+1, 'candidate length invalid')
    require(sum(x is not None for x in pair['C']) <= sum(x is not None for x in pair['H']), 'C capacity exceeds H')

def seal_choices(manifest_path, manifest, refs, requests, needed, receipts, out):
    library = load(refs['candidates088']); seal = load(refs['candidate_seal088'])
    require(library.get('generation_uses_GT') is False and seal.get('sealed_before_GT_evaluation') is True, '088 candidate generation/seal invalid')
    require(seal['candidates_sha256'] == sha(refs['candidates088']), '088 candidate seal SHA mismatch')
    pairs = library['pairs']
    require(set(pairs) == set(needed) and seal['N_pairs'] == len(needed), '088 pair set mismatch')
    audits = load(refs['candidate_audits088'])
    audit_by = {f"{a['video_name']}/{a['pair_idx']}":a for a in audits}
    require(len(audit_by) == len(audits) == len(needed) and set(audit_by) == set(needed), '088 audit pair mismatch')
    choices = {}; score_audits = []
    for key, c in sorted(needed.items()):
        ref = manifest['score_files'][key]; path = reference(manifest_path.parent, ref, receipts)
        require(audit_by[key]['source_sha256'] == ref['sha256'], '088/089 native raw SHA differs')
        n = int(c['N_valid']); frames = int(c['origin'])+4*np.arange(n)
        with np.load(path, allow_pickle=False) as z:
            require(z['native_classes'].shape == (9,51) and z['native_masks'].shape == (9,n), 'native array shape invalid')
            require(np.array_equal(z['frames'], frames), 'native frame coordinates invalid')
            require(np.isfinite(z['native_masks']).all(), 'nonfinite native masks')
            mask_sha = hashlib.sha256(np.ascontiguousarray(z['native_masks']).tobytes()).hexdigest()
            require(mask_sha == audit_by[key]['native_masks_sha256'], '088 mask array SHA differs')
            audit_candidates(pairs[key], c, frames)
            video, pi = key.rsplit('/',1)
            require(pairs[key]['video_name'] == video and pairs[key]['pair_idx'] == int(pi), 'candidate pair identity differs')
            for selection_key, req in sorted(requests.items()):
                if req['pair_key'] == key:
                    choices[selection_key] = dict(pair_key=key, **select_queries(z['native_classes'], req['target_pid'], req['legacy_query']))
        score_audits.append(dict(pair_key=key, source_path=ref['path'], read_path=str(path), source_sha256=ref['sha256'], native_masks_sha256=mask_sha, candidate_count={a: sum(x is not None for x in pairs[key][a]) for a in ('H','C')}))
    write(out/'query_selection.json', dict(selection_receives_GT_time=False, selections=choices))
    choice_seal = dict(sealed_before_GT_evaluation=True, selection_sha256=sha(out/'query_selection.json'), candidates088_sha256=sha(refs['candidates088']), N_selections=len(choices), N_pairs=len(needed))
    write(out/'selection_seal.json', choice_seal); write(out/'pair_audits.json', score_audits)
    return load(out/'query_selection.json')['selections'], pairs, choice_seal

def evaluate(events, needed, bykey, choices, pairs):
    rows = []
    for e in events:
        r = bykey[(e['video_name'], e['gt_index'])]; c = needed[e['pair_key']]
        a,b = int(r['begin_fid']), int(r['end_fid']); fps = float(r['fps'])
        require(a < b and fps > 0 and (b-a)/fps < 3 and float(r['coexist_sec']) >= 10, 'cohort durations changed')
        require(r['chosen_pair_covers_gt'].lower() == 'true' and int(c['pair_start']) <= a < b <= int(c['pair_end']), 'fixed pair no longer covers GT')
        require(int(c['subject_tid']) == int(r['subject_tid']) and int(c['object_tid']) == int(r['object_tid']), 'fixed pair entity mismatch')
        if e['subsecond163']:
            require((b-a)/fps < 1, 'subsecond definition changed')
        choice = choices[e['selection_key']]
        row = dict(e, begin_fid=a, end_fid=b, fps=fps, **evaluate_event(choice, pairs[e['pair_key']], a, b))
        for policy in ('F','T'):
            q = choice[policy]
            row[policy+'_class'] = choice['per_query'][q] if q is not None else None
        for arm in ('H','C'):
            require(row['F_'+arm+'_hit'] <= row['U_'+arm+'_hit'] and row['T_'+arm+'_hit'] <= row['U_'+arm+'_hit'], 'oracle containment failure')
        rows.append(row)
    return rows

def csv_rows(path, rows):
    keys = sorted(set().union(*(r.keys() for r in rows)))
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys); writer.writeheader()
        writer.writerows({k: json.dumps(v, sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows)

def run(manifest_path, out, protocol, software):
    begin = time.monotonic(); out = Path(out).resolve()
    require(not out.exists(), 'fresh output directory required'); out.mkdir(parents=True)
    phase = 'FROZEN_INPUT_CHECKS'
    try:
        manifest_path, protocol, software = (Path(p).resolve() for p in (manifest_path, protocol, software))
        receipts = {}; freeze_checks(manifest_path, protocol, software, receipts)
        manifest, refs, requests, events, needed, bykey = bind(manifest_path, receipts)
        phase = 'CLASS_ONLY_SELECTION_SEAL'
        choices, pairs, seal = seal_choices(manifest_path, manifest, refs, requests, needed, receipts, out)
        print('089_QUERY_SELECTION_SEALED', seal['N_selections'], flush=True)
        phase = 'SEALED_GT_EVALUATION'
        rows = evaluate(events, needed, bykey, choices, pairs)
        require(len(rows) == 400 and sum(r['subsecond163'] for r in rows) == 163, 'evaluation denominator changed')
        full, fi = summarize(rows)
        common, ci = summarize([r for r in rows if r['T_exists']])
        sub, _ = summarize([r for r in rows if r['subsecond163']], bootstrap=False)
        old = load(refs['summary088']); require(old['complete'] is True, '088 result incomplete')
        for group, new in (('primary400',full), ('subsecond163',sub)):
            for arm in ('H','C'):
                require(new['integer_counts']['U_'+arm] == old[group]['integer_counts'][arm+'_hits'], '088 U conservation failure '+group+arm)
        drift = sum(r['score_drift'] for r in rows)
        selection_counts = dict(T_exists_events=sum(r['T_exists'] for r in rows), T_exists_videos=common['N_videos'],
            F_equals_T=sum(choices[r['selection_key']]['F'] == choices[r['selection_key']]['T'] for r in rows),
            F_differs_T_when_T_exists=sum(r['T_exists'] and choices[r['selection_key']]['F'] != choices[r['selection_key']]['T'] for r in rows),
            T_absent=sum(not r['T_exists'] for r in rows),
            T_background_dominant=sum(bool(r['T_class'] and r['T_class']['background_dominates_foreground']) for r in rows),
            unit='per-GT conditional readout opportunities, not independent predictions')
        decision = decide(full, common, drift)
        write(out/'per_gt.json', rows); csv_rows(out/'per_gt.csv', rows)
        write(out/'all_new_losses.json', {a:[r for r in rows if r[a+'_loss']] for a in ('H','C')})
        write(out/'all_new_gains.json', {a:[r for r in rows if r[a+'_gain']] for a in ('H','C')})
        write(out/'per_video.json', dict(primary400=full['per_video'], T_exists=common['per_video'], subsecond163=sub['per_video']))
        np.savez(out/'bootstrap_indices.npz', primary400=fi, T_exists=ci)
        audit = dict(N_GT=400, N_subsecond=163, N_videos=23, N_pairs=len(needed), rank_drift_events=drift,
            independently_recomputed_class_ranks=True, GT_exclusions=0, T_exists_includes_empty_intervals=True,
            oracle_conservation_with088=True, candidate_source_SHA_matches=True, model_forwards=0, training_steps=0)
        write(out/'audits.json', audit)
        phase = 'FINAL_HASH_CHECKS'
        for path, digest in receipts.items():
            require(sha(path) == digest, 'source changed during run '+path)
        require(sha(out/'query_selection.json') == seal['selection_sha256'], 'selection seal changed')
        summary = dict(complete=True, experiment='089', decision=decision, primary400=full, T_exists=common,
            subsecond163=sub, rank_drift_events=drift, selection_counts=selection_counts, selection_seal=seal, source_sha256=receipts,
            scope='Known target predicate and legacy fixed pair; one query per F/T; U is nine-query time oracle, not a capacity-matched gain.',
            interpretation='Residual interval deficit does not identify training, representation, threshold, mask generation or selector causality.',
            bootstrap_seed=89, bootstrap_repetitions=10000, numpy_version=np.__version__,
            model_forwards=0, training_steps=0, independent_review='PENDING', elapsed_seconds=time.monotonic()-begin)
        write(out/'summary.json', summary); write(out/'source_sha256.json', receipts)
        write(out/'status.json', dict(complete=True, status=decision, technical_status='PASS'))
        write(out/'artifact_sha256.json', {p.name:sha(p) for p in out.iterdir() if p.is_file()})
        print('COMPLETE089', decision, flush=True)
    except BaseException as exc:
        write(out/'status.json', dict(complete=False, status='INCONCLUSIVE', technical_status='TECHNICAL_FAILURE', phase=phase, error=repr(exc), elapsed_seconds=time.monotonic()-begin))
        raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--input-manifest', type=Path, default=HERE/'input_manifest.json')
    parser.add_argument('--protocol-freeze', type=Path, default=HERE/'protocol_freeze.json')
    parser.add_argument('--software-freeze', type=Path, default=HERE/'software_freeze.json')
    args = parser.parse_args()
    run(args.input_manifest, args.out, args.protocol_freeze, args.software_freeze)
