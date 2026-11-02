"""Register actually imported 073 evidence; append-only state records."""
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
INCOMING = ROOT / 'handoff/incoming/VIDOR-EVENT-SUPPORT-073'
REMOTE = INCOMING / 'remote_v1'
EID = 'EV-VIDOR-EVENT-SUPPORT-073'


def read(p):
    return json.loads(p.read_text())


def relative(p):
    return str(p.relative_to(ROOT))


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def append(path, key, record):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if any(row.get(key) == record[key] for row in rows):
        raise RuntimeError(f'Already registered: {record[key]}')
    with path.open('a') as stream:
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')


def main():
    checks = read(INCOMING / 'IMPORT_CHECKS.json')
    assert checks['status'] == 'PASS'
    census = read(REMOTE / 'census_v1/summary.json')
    summary = read(REMOTE / 'history_evaluation_v1/summary.json')
    plan = read(REMOTE / 'history_plan_v2/selection.json')
    seal = read(REMOTE / 'history_raw_v2/raw_seal.json')
    budget = read(HERE / 'budget_supplement/results_v2/summary.json')
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    files = [HERE / 'PROTOCOL.md', HERE / 'REPORT.md', HERE / 'history_probe.py',
             HERE / 'analyze_history.py', HERE / 'import_checks.py', HERE / 'census.py',
             HERE / 'budget_supplement/analyze.py', HERE / 'budget_supplement/results_v2/summary.json',
             HERE / 'budget_supplement/results_v2/closure057_reconciliation.json',
             INCOMING / 'IMPORT_CHECKS.json', REMOTE / 'census_v1/summary.json',
             REMOTE / 'history_plan_v2/selection.json', REMOTE / 'history_raw_v2/raw_seal.json',
             REMOTE / 'history_evaluation_v1/summary.json',
             ROOT / 'reviews/event_support_073_code_review.md',
             ROOT / 'reviews/event_support_073_result_review.md']
    result = {
        'evidence_id': EID, 'created_at': stamp,
        'type': 'OBSERVATION_PRIOR_ORACLE_REANALYSIS_AND_FROZEN_WEIGHT_FEATURE_INTERVENTION',
        'status': 'COMPLETE_IMPORTED_COMMANDER_VERIFIED_HISTORY_INDEPENDENT_REVIEW_PASS_WITH_CONCERNS',
        'report': relative(HERE / 'REPORT.md'),
        'remote_path': '/root/autodl-tmp/VrdONE-data/commander_event_support_073',
        'source_evidence_ids': ['EV-VIDOR-NATIVE-THRESHOLD-047', 'EV-VIDOR-NATIVE-THRESHOLD-048', 'EV-VIDOR-NUMERIC-CLOSURE-057'],
        'population': {'definition': 'GT relation_sec<3 and joint_local_continuous_sec>=10; annotated support not verified visibility',
                       'all_GT': 30142, 'target_GT': 3055, 'target_C': 2982, 'target_F': 2050,
                       'target_CF': 1977, 'target_VrdONE_hit100': 67, 'target_BIG_hit100': 14,
                       'detailed_coverage': 331, 'uncovered': 2724,
                       'detailed_target': census['cohorts']['primary_cohort'],
                       'detailed_target_CF': census['cohorts']['primary_CF']},
        'fixed_Top100_opportunity': {'target_misses': 325, 'ANY_grid_feasible_misses': 187,
                                    'grid_opportunity': 69, 'threshold_opportunity': 6, 'default_opportunity': 0,
                                    'Top200_grid_opportunity': 95,
                                    'scope': 'Per-GT oracle, not simultaneous recovery; original047/048 saved source version with057 reconciliation',
                                    'source': relative(HERE / 'budget_supplement/results_v2/summary.json')},
        'history': {'N_cases': plan['N_cases'], 'N_videos': plan['N_videos'], 'N_short_next': plan['N_short_next'],
                    'N_continuing_control_events': 40, 'N_continuing_event_predicate_records': 43,
                    'arms': {'B': 'Original', 'N': 'Earlier no-target-annotation donor',
                             'P': 'Earlier target-annotated donor', 'R': 'Shuffled edited block'},
                    'edit': 'Eight sampled points, old event only; end guard>=.5s and all gap/next/future inputs exact',
                    'primary': summary['primary_all_cases']['primary'],
                    'candidate_existence_not_recall': summary['primary_all_cases']['candidate_existence_not_recall'],
                    'short_candidate_existence_not_recall': summary['short_next_descriptive_only']['candidate_existence_not_recall'],
                    'continuing_predicate_control': summary['primary_all_cases']['continuing_predicate_score_control'],
                    'semantic_causal_claim_allowed': False, 'visual_audit': 'PENDING'},
        'model_forwards': seal['forward_calls'], 'training_steps': 0,
        'GPU_phase_wall_seconds': seal['elapsed_seconds'],
        'new_predictions': '348 sealed query-logit/mask arrays on GT-assisted altered inputs, not official relation detections',
        'formal_mAP': 'NOT_APPLICABLE', 'formal_R_at_100': 'NOT_APPLICABLE',
        'independent_generalization': False,
        'verification': checks,
        'independent_review': {'history_code': 'PASS_WITH_CONCERNS', 'history_results': 'PASS_WITH_CONCERNS', 'budget_supplement': 'PASS_WITH_CONCERNS',
                               'reports': ['reviews/event_support_073_code_review.md', 'reviews/event_support_073_result_review.md']},
        'artifact_hashes': {relative(p): digest(p) for p in files},
        'raw_bundle_sha256': checks['bundle_sha256'],
        'supports': [
            'Among331covered long-coexistence shortGT,148 temporal-output gaps and124 input-infeasible; not a full-population mechanism estimate.',
            '69/325misses have a fixed-native-Top100 grid-time oracle witness; only6 have a saved-query threshold witness.',
            'On87selected events/19videos, both prescribed history effects fall below+.03 score-unit threshold; all-sample intervals include zero.',
            'Short next-event hull+Top6 candidates B/N/P1/28, R2/28; P loses one full-cohort binary+Top6 candidate.'
        ],
        'does_not_support': [
            'Memory never affects VrdONE, all examples are unaffected, or historical semantics were removed.',
            'Gap annotations are verified true negatives, or watch-heavy sample represents hold/release.',
            'Oracle69 equals69 new simultaneous matches, or edited-feature probe improves formal recall/mAP.',
            'Continuous-predicate score control proves preservation under occlusion or formal sustained-relation performance.',
            'A new memory architecture is ready for training or formal paper direction is adopted.'
        ]}
    # Assert identifiers are not already present before any append.
    for name, key, ident in [('evidence.jsonl', 'evidence_id', EID),
                             ('decisions.jsonl', 'decision_id', 'DEC-073-CLOSE-HISTORY-PROBE-PRESERVE-TEMPORAL-GAP')]:
        assert not any(json.loads(l).get(key) == ident for l in (ROOT / 'state' / name).read_text().splitlines() if l.strip())
    append(ROOT / 'state/evidence.jsonl', 'evidence_id', result)
    for endpoint, statement in [
        ('OFF', 'In the frozen0738-point old-event substitution, no-target-annotation history reduces gap target-edge scores relative to target-annotated history by the prescribed positive effect.'),
        ('RESTART', 'In the frozen0738-point old-event substitution, no-target-annotation history improves next-event target-edge scores relative to target-annotated history by the prescribed positive effect.')]:
        append(ROOT / 'state/hypotheses.jsonl', 'hypothesis_id', {
            'hypothesis_id': 'HYP-VIDOR-073-HISTORY-' + endpoint, 'created_at': stamp, 'updated_at': stamp,
            'status': 'BELOW_PRESPECIFIED_EFFECT_CURRENT_OPERATION', 'statement': statement,
            'evidence_ids': [EID], 'scope': '87 exposed GT-assisted selected events/19videos; eight-point feature replacement; not proof all historical effects absent',
            'primary': summary['primary_all_cases']['primary'][endpoint.lower()]})
    append(ROOT / 'state/decisions.jsonl', 'decision_id', {
        'decision_id': 'DEC-073-CLOSE-HISTORY-PROBE-PRESERVE-TEMPORAL-GAP', 'created_at': stamp, 'evidence_ids': [EID],
        'decision': 'Record both history endpoints below the frozen effect threshold; do not launch retention/expiry/restart training from this operation. Preserve fixed-budget short-event temporal-output gaps as an unresolved question. No routine block-length, threshold or seed search to rescue a failed endpoint. Candidate next audit separates onset, offset, missing peak and bridging in existing fixed-budget opportunities while retaining sustained-relation controls.',
        'formal_direction_adoption': False, 'formal_method_adoption': False, 'training_launched': False,
        'remote_job_active': False})
    (HERE / 'ledger.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'registered': EID, 'created_at': stamp, 'report': result['report']}))


if __name__ == '__main__':
    main()
