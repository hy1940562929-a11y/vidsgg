#!/usr/bin/env python3
"""083: deterministic decomposition of already imported temporal oracles.

No model imports, training, inference or new query selection. GT-time oracles
remain diagnostic bounds across all continuous segments of the same identity.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import math
import sys
import tarfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REF = ROOT / 'analysis/temporal_support_015/deployment/reference_per_gt.csv'
ARCHIVE = ROOT / 'handoff/incoming/VIDOR-TEMPORAL-SUPPORT-015/diagnostic_evidence_v1.tar.gz'
MEMBER = 'diagnosis_v1/per_gt.csv'
JOINED = ROOT / 'analysis/week1_short_relation_diag_080/results_v1/per_gt_boundary_joined.csv'
EXPECTED = {'reference': '63a226883d9efe27f8b05aa290b6fd6642119b65bb5178725ce9886dadf53239',
            'diagnosis_member': '7f18e383778cc8d9cd86c90ff00e0653f1d028e24a874ae376f47f0348400a6a',
            'joined': '646841df861a2f21e86076165b86ff6c96e1d90b7233282708930df594df8435'}
STAGES = ('default_hit', 'grid_unfiltered_infeasible', 'grid_min5_infeasible',
          'default_min5_rejected', 'sigmoid_threshold_reachable',
          'raw_logit_only_reachable', 'grid_feasible_threshold_family_infeasible')
NAMES = ('默认查询已有匹配时间形状', '网格无长度限制仍不可行', '网格受最短5帧限制',
         '默认匹配hull被最短5帧过滤', '改sigmoid阈值上界可达', '只有raw-logit阈值上界可达',
         '网格可行但全部阈值hull族不可达')
THRESHOLDS = (.3, .5, .7)
SHAPES = ('default_min5', 'default_unfiltered', 'grid_min5', 'grid_unfiltered', 'sigmoid', 'raw_logit')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def rows(data):
    return list(csv.DictReader(io.StringIO(data.decode('utf-8-sig'))))


def key(row):
    return row['video_name'], int(row['gt_index'])


def index(records):
    out = {}
    for r in records:
        k = key(r)
        require(k not in out, 'Duplicate GT: '+str(k))
        out[k] = r
    return out


def boolean(value):
    require(str(value).lower() in ('true', 'false'), 'Invalid bool: '+str(value))
    return str(value).lower() == 'true'


def number(value):
    result = float(value)
    require(math.isfinite(result), 'Nonfinite numeric value')
    return result


def validate_bounds(r):
    for f in SHAPES:
        require(0 <= r[f] <= 1, 'Out of range '+f)
    # Same identity/pair/query universe and first-last hull grid, checked below.
    pairs = [('default_min5', 'default_unfiltered'), ('default_min5', 'sigmoid'),
             ('sigmoid', 'raw_logit'), ('raw_logit', 'grid_min5'),
             ('grid_min5', 'grid_unfiltered'), ('default_unfiltered', 'grid_unfiltered')]
    for small, large in pairs:
        require(r[small] <= r[large] + 1e-7, f'Oracle hierarchy violation: {small}>{large}')


def stage(r, threshold):
    validate_bounds(r)
    if r['default_min5'] >= threshold:
        return STAGES[0]
    if r['grid_unfiltered'] < threshold:
        return STAGES[1]
    if r['grid_min5'] < threshold:
        return STAGES[2]
    if r['default_unfiltered'] >= threshold:
        return STAGES[3]
    if r['sigmoid'] >= threshold:
        return STAGES[4]
    if r['raw_logit'] >= threshold:
        return STAGES[5]
    return STAGES[6]


def load_records():
    contents = {'reference': REF.read_bytes(), 'joined': JOINED.read_bytes()}
    with tarfile.open(ARCHIVE, 'r:gz') as archive:
        matches = [m for m in archive.getmembers() if m.name == MEMBER]
        require(len(matches) == 1 and matches[0].isfile(), 'Invalid archive member')
        contents['diagnosis_member'] = archive.extractfile(matches[0]).read()
    for name, data in contents.items():
        require(digest(data) == EXPECTED[name], 'Source SHA mismatch: '+name)
    a, b, c = (index(rows(contents[name])) for name in ('reference', 'diagnosis_member', 'joined'))
    require(set(a) == set(b) == set(c) and len(c) == 2573, 'GT source coverage mismatch')
    records = []
    for k, joined in sorted(c.items()):
        ref, diag = a[k], b[k]
        for col in ('subject_tid', 'object_tid', 'begin_fid', 'end_fid'):
            require(int(ref[col]) == int(diag[col]) == int(joined[col]), f'GT field mismatch {k} {col}')
        for col in ('subject_class', 'object_class', 'predicate'):
            require(ref[col] == diag[col] == joined[col], f'GT label mismatch {k} {col}')
        require(ref['group_at100'] == diag['group_at100'] == joined['original_group'], 'Original group mismatch')
        fps = number(joined['fps'])
        duration = (int(joined['end_fid'])-int(joined['begin_fid'])) / fps
        require(fps > 0 and duration > 0, 'Invalid GT duration')
        for value in (ref['duration_sec'], diag['duration_sec'], joined['relation_sec']):
            require(abs(number(value)-duration) < 1e-7, 'Duration mismatch')
        r = {col: joined[col] for col in ('video_name', 'gt_index', 'subject_tid', 'object_tid', 'predicate',
                                          'begin_fid', 'end_fid', 'coexist_sec', 'original_group')}
        r.update(relation_sec=duration, fps=fps,
                 chosen_pair_covers_gt=boolean(joined['chosen_pair_covers_gt']),
                 selected_tiou=number(joined['tiou']),
                 default_min5=number(diag['default_hull_tiou_min5']),
                 default_unfiltered=number(diag['default_hull_tiou_unfiltered']),
                 grid_min5=number(ref['grid_hull_tiou_upper_min5']),
                 grid_unfiltered=number(ref['grid_hull_tiou_upper_unfiltered']),
                 sigmoid=number(diag['any_query_threshold_oracle_tiou']),
                 raw_logit=number(diag['any_query_raw_logit_threshold_oracle_tiou']),
                 raw_queries=int(diag['all_correct_identity_raw_queries']),
                 same_identity_pairs=int(diag['correct_identity_pair_count']))
        require(r['default_min5'] == number(joined['oracle_tiou']) == number(ref['gt_tiou50_min5_best']),
                '015 vs 077 default oracle mismatch')
        require(r['default_unfiltered'] == number(ref['gt_tiou50_unfiltered_best']), 'Unfiltered oracle mismatch')
        require(r['grid_min5'] == number(diag['grid_hull_upper_min5_reference']), 'Grid reference mismatch')
        require(r['selected_tiou'] <= r['default_min5'] + 1e-7, 'Selected above oracle')
        validate_bounds(r)
        r['duration_group'] = '<1s' if duration < 1 else '1-3s' if duration < 3 else '>=3s'
        for threshold in THRESHOLDS:
            tag = str(int(threshold * 10))
            r['stage_tiou0'+tag] = stage(r, threshold)
            r['selected_partition_tiou0'+tag] = ('selected_hit' if r['selected_tiou'] >= threshold else
                'oracle_only' if r['default_min5'] >= threshold else 'neither')
        records.append(r)
    require(len({r['video_name'] for r in records}) == 27, 'Wrong video cohort')
    return records, {name: {'sha256': digest(data), 'bytes': len(data)} for name, data in contents.items()}


def summarize(records):
    result = []
    for population in ('all', 'chosen_pair_covers_gt'):
        pop = [r for r in records if population == 'all' or r['chosen_pair_covers_gt']]
        for duration in ('all', '<1s', '1-3s', '>=3s'):
            group = [r for r in pop if duration == 'all' or r['duration_group'] == duration]
            for threshold in THRESHOLDS:
                counts = Counter(stage(r, threshold) for r in group)
                require(sum(counts.values()) == len(group), 'Partition does not close')
                out = dict(population=population, duration_group=duration, tiou_threshold=threshold,
                           N=len(group), N_videos=len({r['video_name'] for r in group}))
                for name in STAGES:
                    out[name+'_numerator'] = counts[name]
                    out[name+'_denominator'] = len(group)
                    out[name+'_fraction'] = counts[name]/len(group) if group else None
                result.append(out)
    return result


def write_csv(path, records):
    with path.open('x', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0])); writer.writeheader()
        writer.writerows({k: 'NA' if v is None else v for k, v in r.items()} for r in records)


def report(summary):
    text = ['# 083：已有时间输出的可行性分解', '',
        '只重新统计已导入的015/077结果，不训练、不推理。所有oracle跨同身份的全部连续实体段和全部query，不要求谓词正确；数字不是新增正确检测数。', '',
        '|人群|N|默认已有|网格不行|网格min5不行|默认min5拒绝|sigmoid阈值可达|仅logit可达|全部阈值族不可达|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        if r['tiou_threshold'] != .5: continue
        label = ('全部' if r['population']=='all' else '所选实体段覆盖GT')+' / '+r['duration_group']
        text.append('|'+label+'|'+str(r['N'])+'|'+'|'.join(str(r[s+'_numerator']) for s in STAGES)+'|')
    focus = next(r for r in summary if r['population']=='chosen_pair_covers_gt' and r['duration_group']=='<1s' and r['tiou_threshold']==.5)
    text += ['', '表中七类互斥，相加等于各行N；计数顺序与冻结协议一致。全部比例与0.3/0.5/0.7扫描在CSV，未筛除无输出GT。', '', '**已知后选择的亚秒焦点**：']
    for s, name in zip(STAGES, NAMES):
        n = focus[s+'_numerator']; den = focus['N']
        text.append(f'- {name}：{n}/{den} = {100*n/den:.2f}%。')
    text += ['', '**解释边界**：网格可行只说明首末格点容得下区间，不能排除真实采样或时序特征不足。阈值上界不可达说明现有分数排序与hull解码族不足，不能单独定位到训练、特征或历史干扰。min5与阈值类别按优先级分解；原独立数值保留在逐GT表。', '',
             '015默认oracle与077默认oracle逐GT精确一致，数据集合2,573条/27视频完全一致。没有实际模型恢复数，没有新的mAP/R@K，也不改变080的机制未决判决。']
    return '\n'.join(text)+'\n'


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), 'Refuse existing output')
    protocol = json.loads((HERE/'protocol_freeze.json').read_text())
    require(digest((HERE/'PROTOCOL.md').read_bytes()) == protocol['protocol_sha256'], 'Protocol changed after freeze')
    records, sources = load_records()
    summary = summarize(records)
    args.out.mkdir(parents=True)
    write_csv(args.out/'per_gt.csv', records); write_csv(args.out/'summary.csv', summary)
    (args.out/'REPORT.md').write_text(report(summary), encoding='utf-8')
    manifest = dict(complete=True, created_at=datetime.now(timezone.utc).isoformat(),
        analysis='083_existing_oracles_posthoc_decomposition', N_GT=len(records), N_videos=27,
        model_forwards=0, training_steps=0, checkpoint_loaded=False, sources=sources,
        source_paths=dict(reference=str(REF), diagnosis_archive=str(ARCHIVE), diagnosis_member=MEMBER, joined=str(JOINED)),
        protocol_sha256=protocol['protocol_sha256'], code_sha256=digest(Path(__file__).read_bytes()),
        output_sha256={p.name: digest(p.read_bytes()) for p in sorted(args.out.iterdir())},
        limits=['GT-time oracle, any predicate, all same-identity continuous segments', 'not actual recovery',
                'posthoc focus from 081, not independent validation'])
    (args.out/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(complete=True, N_GT=len(records), summary_rows=len(summary), output=str(args.out))))


if __name__ == '__main__':
    main()
