"""Audio-free rederivation of the adversarial-review evidence.

This checks retained predictions, gates, physical-grid/PCM hash ledgers,
metadata strata and replica ranges. It does not decode audio, rebuild caches,
rerun retrieval, or establish a new device/production measurement.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
import math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from derive_distortion_strata import derive as derive_strata
from run_acr_protocol import query_crop
from run_digital_boundary import CONDITIONS, SCIENCE_FIELDS
from summarize_results import cluster_interval, crossed_interval, select_threshold, wilson
from verify_v2_evidence import load, original_sha, sha

ROOT = Path(__file__).resolve().parents[2]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def same(actual, expected, label):
    require(actual == expected, f"{label} differs")


def bind(path, expected, root):
    same(original_sha(path, root), expected, f"Original execution SHA: {path.name}")


def check_code(plan):
    for name, digest in plan['source_sha256'].items():
        same(sha(Path(__file__).parent / name), digest, f"Frozen source {name}")


def cohort(protocol):
    rows = [query_crop(q, 5) for q in protocol['queries'] if q['evaluate']]
    require(len(rows) == 835 and len({q['query_id'] for q in rows}) == 835, 'Original query cohort')
    gallery = [r for r in protocol['references'] if r['role'] in ('calibration_known', 'test_known')]
    require(len(gallery) == 659, 'Original gallery cohort')
    return rows, gallery


def validate_row(row, query, gallery, expected_offset, duration):
    for field in ('source_id', 'source_query_id', 'role'):
        same(row[field], query[field], f"Query {row['query_id']} {field}")
    same(row['target_reference_id'], query['reference_id'], 'Target reference')
    require(row['reference_id'] is None or row['reference_id'] in gallery, 'Candidate outside gallery')
    same(row['expected_reference_start_s'], expected_offset, 'Physical ground truth')
    same(row['duration_s'], duration, 'Input duration')
    require(math.isfinite(row['score']), 'Nonfinite score')
    correct = row['reference_id'] == query['reference_id']
    localized = bool(correct and abs(row['offset_s'] - expected_offset) <= 2.)
    same(row['correct_reference'], correct, 'Stored content label')
    same(row['localized_correct'], localized, 'Stored localization label')
    if row['reference_id'] is not None:
        same(row['score'], row['vote_fraction'] / (1 + row['mean_squared_l2']), 'Score arithmetic')
    return correct, localized


def verify_boundary(results, review, root):
    directory = review / 'digital_boundary'
    plan_path = directory / 'plan.json'
    plan, summary = load(plan_path), load(directory / 'summary.json')
    bind(results / 'hard_medium_protocol.json', plan['protocol_sha256'], root)
    bind(results / 'acr_medium/pca_model.npz', plan['pca_model_sha256'], root)
    bind(results / 'acr_ivf_efficiency_study/acr_ivf_selected_5s.json', plan['frozen_result_sha256'], root)
    plan_sha = original_sha(plan_path, root)
    same(sha(Path(__file__).with_name('run_digital_boundary.py')), plan['runner_sha256'], 'Stress runner')
    check_code(plan)
    same(plan['conditions'], list(CONDITIONS), 'All four prescribed transforms')
    require(plan['all_conditions_mandatory'] and plan['condition_selection'] is None, 'Outcome-selected stress condition')
    require(not any(plan[k] for k in ('pca_refit', 'centroid_retraining', 'gate_retuning', 'timing_claim')), 'Changed frozen stress scope')
    same(plan['matcher'], {'reference_factor': 8, 'query_stride': 4, 'ivf_nlist': 512,
                          'nprobe': 16, 'top_k': 5, 'tolerance_sec': .12, 'threads': 3}, 'Frozen stress matcher')
    same(summary['plan_sha256'], plan_sha, 'Stress summary plan')
    protocol = load(results / 'hard_medium_protocol.json')
    queries, gallery_rows = cohort(protocol)
    qmap = {q['query_id']: q for q in queries}
    gallery = {r['reference_id'] for r in gallery_rows}
    same(plan['query_ids'], list(qmap), 'Stress query ordering')
    same(plan['gallery_reference_ids'], [r['reference_id'] for r in gallery_rows], 'Stress gallery ordering')
    roles_expected = Counter(q['role'] for q in queries)
    same(plan['query_role_counts'], dict(roles_expected), 'Stress role denominators')
    frozen = load(results / 'acr_ivf_efficiency_study/acr_ivf_selected_5s.json')
    previous = {r['query_id']: r for r in frozen['predictions']}
    gate = select_threshold([r for r in previous.values() if r['role'] == 'calibration_unknown'], .01)
    same(gate, frozen['frozen_threshold'], 'Original global gate')
    same(gate, plan['frozen_original_5s_gate'], 'Stress frozen global gate')
    ledger = load(directory / 'decoded_input_ledger.json')
    same(ledger['plan_sha256'], plan_sha, 'Decoded ledger plan')
    same(set(ledger['original_crops']), set(qmap), 'Crop ledger coverage')
    same(set(ledger['source_audio']), set(plan['source_audio_sha256']), 'Source ledger coverage')
    require(len(ledger['source_audio']) == 219, 'Source montage count')
    for path, record in ledger['source_audio'].items():
        same(record['file_sha256'], plan['source_audio_sha256'][path], 'Declared source bytes')
        same(record['sample_rate'], 8000, 'Decoded ledger sample rate')
        require(record['frames'] > 0 and len(record['decoded_pcm_sha256']) == 64, 'Decoded ledger geometry/digest')
    for qid, record in ledger['original_crops'].items():
        query = qmap[qid]
        same(record['path'], query['path'], 'Crop source path')
        same(record['start_frame'], round(query['start_s'] * 8000), 'Crop start')
        same(record['end_frame'], record['start_frame'] + 40000, 'Crop endpoint')
        same(record['frames'], 40000, 'Original crop frames')
        require(len(record['crop_pcm_sha256']) == 64, 'Crop PCM digest')
    baseline = None
    derived, switches = [], []
    for condition, reported in zip(CONDITIONS, summary['rows'], strict=True):
        path = directory / f"{condition['name']}.json"
        result = load(path)
        bind(path, reported['result_sha256'], root)
        same(result['condition'], condition, 'Stress transform')
        same(result['plan_sha256'], plan_sha, 'Stress result plan')
        same(result['protocol_sha256'], plan['protocol_sha256'], 'Stress result protocol')
        same(result['frozen_threshold'], gate, 'Stress result gate')
        require(result['gate_retuned'] is False and result['timing_claim'] is False, 'Stress outcome tuning/timing')
        rows = result['predictions']
        same([r['query_id'] for r in rows], plan['query_ids'], 'Complete stress cohort/order')
        same(Counter(r['role'] for r in rows), roles_expected, 'Stress split')
        for r in rows:
            q = qmap[r['query_id']]
            expected = q['expected_reference_start_s'] + (condition['trim_s'] - condition['lead_s']) * q['expected_time_scale']
            correct, localized = validate_row(r, q, gallery, expected, condition['output_s'])
            same(r['original5s_expected_reference_start_s'], q['expected_reference_start_s'], 'Original annotation')
            same(r['expected_time_scale'], q['expected_time_scale'], 'Affine source scale')
            same(r['input_frames'], round(condition['output_s'] * 8000), 'Transformed frame count')
            same(r['leading_zero_frames'], round(condition['lead_s'] * 8000), 'Leading silence frames')
            same(r['trim_frames'], round(condition['trim_s'] * 8000), 'Trimmed frames')
            frames = r['query_physical_frame_indices']
            same(len(frames), r['query_fingerprints'], 'Searched frame count')
            require(frames == sorted(set(frames)) and all(0 <= i <= r['input_frames']//186-31 and i % 4 == 0 for i in frames), 'Original physical query residues')
            same(r['empty_features'], len(frames) == 0, 'Empty feature label')
            require(len(r['transformed_pcm_sha256']) == 64, 'Transformed PCM digest')
            accepted = r['reference_id'] is not None and r['score'] >= gate
            strict = bool(correct and abs(r['offset_s'] - expected) <= .1)
            expected_labels = {'accepted': accepted, 'accepted_correct': accepted and correct,
                               'accepted_localized': accepted and localized, 'accepted_wrong': accepted and not correct,
                               'localized_diagnostic_0p1s': strict, 'accepted_localized_diagnostic_0p1s': accepted and strict}
            for key, value in expected_labels.items(): same(r[key], value, f'Stress label {key}')
            if condition['name'] == 'unchanged_5s':
                for field in SCIENCE_FIELDS: same(r[field], previous[r['query_id']][field], f'835-row unchanged conformance {field}')
                same(r['transformed_pcm_sha256'], ledger['original_crops'][r['query_id']]['crop_pcm_sha256'], 'Unchanged PCM bytes')
        if baseline is None: baseline = {r['query_id']: r for r in rows}
        role_counts = {}
        for role in roles_expected:
            sub = [r for r in rows if r['role'] == role]
            role_counts[role] = {'n': len(sub), 'candidate_correct': sum(r['correct_reference'] for r in sub),
              'candidate_localized_2s': sum(r['localized_correct'] for r in sub),
              'accepted_correct': sum(r['accepted_correct'] for r in sub), 'accepted_localized_2s': sum(r['accepted_localized'] for r in sub),
              'accepted_wrong_id': sum(r['accepted_wrong'] for r in sub), 'accepted': sum(r['accepted'] for r in sub),
              'no_candidate': sum(r['reference_id'] is None for r in sub), 'empty_features': sum(r['empty_features'] for r in sub),
              'diagnostic_0p1s_candidate': sum(r['localized_diagnostic_0p1s'] for r in sub),
              'diagnostic_0p1s_accepted': sum(r['accepted_localized_diagnostic_0p1s'] for r in sub)}
        same(role_counts, reported['roles'], 'All stress counts')
        known = [r for r in rows if r['role'] == 'test_known']
        unknown = [r for r in rows if r['role'] == 'test_unknown']
        for metric, intervals in reported['test_known_uncertainty'].items():
            same({'source_cluster': cluster_interval(known, metric), 'source_montage_crossed': crossed_interval(known, metric)}, intervals, 'Stress clustered intervals')
        for metric, intervals in reported['paired_difference_vs_unchanged'].items():
            diffs = [{'source_id': r['source_id'], 'source_query_id': r['source_query_id'],
                      'difference': int(r[metric]) - int(baseline[r['query_id']][metric])} for r in known]
            same({'source_cluster': cluster_interval(diffs, 'difference'), 'source_montage_crossed': crossed_interval(diffs, 'difference')}, intervals, 'Stress paired intervals')
        udiff = [{'source_id': r['source_id'], 'source_query_id': r['source_query_id'],
                  'difference': int(r['accepted']) - int(baseline[r['query_id']]['accepted'])} for r in unknown]
        false_count = sum(r['accepted'] for r in unknown)
        same({'n': len(unknown), 'false_accepts': false_count, 'wilson_ci95': wilson(false_count, len(unknown)),
              'source_cluster': cluster_interval(unknown, 'accepted'), 'source_montage_crossed': crossed_interval(unknown, 'accepted'),
              'paired_source_montage_crossed': crossed_interval(udiff, 'difference')}, reported['unknown_false_accepts'], 'Unknown stress intervals')
        switch = {'condition': condition['name'], 'wrong_candidate_known_ids': sum(not r['correct_reference'] and r['reference_id'] is not None for r in known)}
        for label, metric in [('candidate_correctness', 'correct_reference'), ('accept_correct', 'accepted_correct'), ('accepted_localized', 'accepted_localized')]:
            switch[label+'_lost'] = sum(baseline[r['query_id']][metric] and not r[metric] for r in known)
            switch[label+'_gained'] = sum(not baseline[r['query_id']][metric] and r[metric] for r in known)
        for label, sub in [('all835', rows), ('test_known', known)]:
            changed = sum(r['reference_id'] != baseline[r['query_id']]['reference_id'] for r in sub)
            switch['ids_changed_'+label] = changed
            same(changed, reported['candidate_ids_changed_'+label], 'Stress changed IDs')
        same({'min': min(r['query_fingerprints'] for r in known), 'median': float(np.median([r['query_fingerprints'] for r in known])),
              'max': max(r['query_fingerprints'] for r in known)}, reported['query_fingerprints_test_known'], 'Stress query counts')
        switches.append(switch)
        derived.append({'condition': condition['name'], 'roles': role_counts})
    same(switches, load(directory / 'decision_switches.json')['rows'], 'All stress decision switches')
    parity = load(directory / 'unchanged_parity.json')
    same(parity['query_count'], 835, 'Fresh unchanged audit count')
    same(parity['plan_sha256'], plan_sha, 'Fresh unchanged audit plan')
    return {'conditions': 4, 'prediction_rows': 3340, 'unchanged_exact_parity_rows': 835,
            'all_frozen_gates_geometry_counts_intervals_switches_rederived': True,
            'pcm_digests_structurally_checked': True, 'pcm_redecoded_in_this_check': False, 'rows': derived}


def verify_phases(results, review, root):
    directory = review / 'reference_phase'
    plan_path = directory / 'plan.json'
    plan, strict, residues = load(plan_path), load(directory / 'strict_validation.json'), load(directory / 'reference_residue_counts.json')
    bind(results / 'hard_medium_protocol.json', plan['protocol_sha256'], root)
    bind(results / 'acr_medium/pca_model.npz', plan['pca_model_sha256'], root)
    bind(results / 'acr_medium/acr_d8_5s.json', plan['phase0_canonical_result_sha256'], root)
    bind(results / 'acr_medium/acr_d1_5s.json', strict['dense_baseline_sha256'], root)
    plan_sha = original_sha(plan_path, root)
    same(sha(Path(__file__).with_name('run_reference_phase.py')), plan['runner_sha256'], 'Phase runner')
    check_code(plan)
    same(plan['phases'], list(range(8)), 'Complete phase plan')
    require(plan['phase_selection'] is None and plan['all_phases_mandatory'] and not plan['pca_refit'] and not plan['timing_claim'], 'Changed phase scope')
    same(plan['factor'], 8, 'Phase factor')
    same(plan['duration_s'], 5., 'Phase requested duration')
    same(plan['matcher'], {'exact': True, 'top_k': 5, 'tolerance_sec': .12,
                          'localization_tolerance_s': 2., 'threads': 3,
                          'score': 'vote_fraction/(1+mean_squared_L2)'}, 'Frozen exact phase matcher')
    same(residues['plan_sha256'], plan_sha, 'Residue plan identity')
    bind(directory / 'reference_residue_counts.json', strict['reference_residue_ledger_sha256'], root)
    protocol = load(results / 'hard_medium_protocol.json')
    queries, references = cohort(protocol)
    qmap = {q['query_id']: q for q in queries}
    gallery = {r['reference_id'] for r in references}
    same(plan['query_ids'], list(qmap), 'Phase query order')
    same(plan['gallery_reference_ids'], [r['reference_id'] for r in references], 'Phase gallery order')
    same([r['reference_id'] for r in residues['rows']], [r['reference_id'] for r in references], 'Residue source coverage/order')
    cache_hashes = list(plan['cache_sha256'].values())
    require(len(cache_hashes) == 1494, 'Frozen source/query cache coverage')
    totals = np.zeros(8, dtype=np.int64)
    total_duration = 0.
    for row, reference, cache_digest in zip(residues['rows'], references, cache_hashes[:659], strict=True):
        same(row['role'], reference['role'], 'Residue source role')
        same(row['feature_cache_sha256'], cache_digest, 'Residue source digest bound to ordered frozen gallery')
        counts = row['original_residue_counts']
        require(len(counts) == 8 and all(isinstance(n, int) and n >= 0 for n in counts), 'Residue counts')
        same(sum(counts), row['surviving_frames'], 'All surviving frames assigned to one original residue')
        require(row['maximum_grid_reconstruction_residual_s'] <= 1e-8 and
                row['original_frames_nonnegative_unique_strictly_monotonic'] is True, 'Recorded strict physical-grid audit')
        require(row['audio_duration_s'] > 0, 'Reference duration')
        totals += counts
        total_duration += row['audio_duration_s']
    same(totals.tolist(), residues['phase_totals'], 'Residue phase totals')
    require(math.isclose(total_duration, residues['reference_audio_duration_s'], abs_tol=1e-8), 'Summed reference duration')
    phase0 = {r['query_id']: r for r in load(results / 'acr_medium/acr_d8_5s.json')['predictions']}
    dense = load(results / 'acr_medium/acr_d1_5s.json')['predictions']
    dense_map = {r['query_id']: r for r in dense}
    dense_gate = select_threshold([r for r in dense if r['role'] == 'calibration_unknown'], .01)
    rows_derived = []
    for phase in range(8):
        path = directory / f'phase_{phase}_5s.json'
        result = load(path)
        bind(path, strict['rows'][phase]['result_sha256'], root)
        same(result['reference_phase'], phase, 'Declared result phase')
        same(result['plan_sha256'], plan_sha, 'Phase result plan')
        same(result['protocol_sha256'], plan['protocol_sha256'], 'Phase result protocol')
        same(result['matcher'], plan['matcher'], 'Phase matcher')
        same(result['config'], plan['config'], 'Phase encoder configuration')
        same(result['factor'], 8, 'Phase result reference factor')
        same(result['duration_s_requested'], 5., 'Phase result requested duration')
        require(not result['phase_selected'] and not result['timing_claim'], 'Selected phase or timing claim')
        predictions = result['predictions']
        same([r['query_id'] for r in predictions], plan['query_ids'], 'Complete phase cohort/order')
        for r in predictions:
            q = qmap[r['query_id']]
            validate_row(r, q, gallery, q['expected_reference_start_s'], 5.)
            same(r['query_fingerprints'], phase0[r['query_id']]['query_fingerprints'], 'Dense query preserved')
            if phase == 0:
                for field in SCIENCE_FIELDS[:-1]: same(r[field], phase0[r['query_id']][field], 'Phase0 exact canonical conformance')
        same(result['stats']['fingerprints'], int(totals[phase]), 'Physical residue count')
        require(math.isclose(result['stats']['reference_audio_duration_s'], total_duration, abs_tol=1e-6), 'Phase reference duration')
        calibration = [r for r in predictions if r['role'] == 'calibration_unknown']
        known = [r for r in predictions if r['role'] == 'test_known']
        unknown = [r for r in predictions if r['role'] == 'test_unknown']
        same((len(calibration), len(known), len(unknown)), (74, 543, 71), 'Phase denominators')
        gate = select_threshold(calibration, .01)
        accepted = lambda r: r['reference_id'] is not None and r['score'] >= gate
        item = {'phase': phase, 'fingerprints': int(totals[phase]), 'payload_MB_per_hour': totals[phase]*128/total_duration*3600/1e6,
          'threshold': gate, 'calibration_false_accepts': sum(accepted(r) for r in calibration),
          'candidate_correct': sum(r['correct_reference'] for r in known), 'candidate_localized': sum(r['localized_correct'] for r in known),
          'accepted_correct': sum(accepted(r) and r['correct_reference'] for r in known),
          'accepted_localized': sum(accepted(r) and r['localized_correct'] for r in known),
          'accepted_target_mismatch': sum(accepted(r) and not r['correct_reference'] for r in known),
          'unknown_accepts': sum(accepted(r) for r in unknown),
          'known_candidate_id_changes_vs_phase0': sum(r['reference_id'] != phase0[r['query_id']]['reference_id'] for r in known),
          'all_candidate_id_changes_vs_phase0': sum(r['reference_id'] != phase0[r['query_id']]['reference_id'] for r in predictions),
          'raw_paired_difference_vs_dense': crossed_interval([{'source_id': r['source_id'], 'source_query_id': r['source_query_id'],
              'difference': int(r['correct_reference'])-int(dense_map[r['query_id']]['correct_reference'])} for r in known], 'difference'),
          'accepted_paired_difference_vs_dense': crossed_interval([{'source_id': r['source_id'], 'source_query_id': r['source_query_id'],
              'difference': int(accepted(r) and r['correct_reference'])-int(dense_map[r['query_id']]['reference_id'] is not None and dense_map[r['query_id']]['score'] >= dense_gate and dense_map[r['query_id']]['correct_reference'])} for r in known], 'difference'),
          'result_sha256': original_sha(path, root)}
        same(item, strict['rows'][phase], 'All phase counts/gates/paired intervals')
        require(item['calibration_false_accepts'] == 0, 'Original empirical gate constraint')
        rows_derived.append(item)
    fields = ('candidate_correct', 'candidate_localized', 'accepted_correct', 'accepted_localized', 'accepted_target_mismatch', 'unknown_accepts', 'fingerprints', 'payload_MB_per_hour')
    ranges = {field: [min(r[field] for r in rows_derived), max(r[field] for r in rows_derived)] for field in fields}
    same(ranges, strict['phase_ranges'], 'All phase ranges')
    return {'phases': 8, 'predictions_per_phase': 835, 'prediction_rows': 6680, 'all_phase_gates_counts_intervals_rederived': True,
            'residue_ledger_sources': 659, 'feature_caches_reread_in_this_check': False, 'phase_ranges': ranges, 'rows': rows_derived}


def verify_strata(results, review):
    args = SimpleNamespace(protocol=results/'hard_medium_protocol.json', benchmarks=results,
                           paper_numbers=results/'paper_numbers_v2.json', upstream=None)
    derived = derive_strata(args)
    old = load(review / 'distortion_strata.json')
    for field in ('groups', 'query_assignments', 'per_query_outcomes', 'methods', 'group_definitions', 'uncertainty'):
        same(derived[field], old[field], f'All eight configuration strata {field}')
    # Do not rely on a stored localization flag when rederiving the strata.
    protocol = load(args.protocol)
    qmap = {q['query_id']: query_crop(q, 5) for q in protocol['queries'] if q['evaluate']}
    for method in old['methods'].values():
        for row in load(results / method['prediction_path'])['predictions']:
            query = qmap[row.get('base_query_id', row['query_id'].split('@')[0])]
            expected = query['expected_reference_start_s']
            localized = bool(row['reference_id'] == query['reference_id'] and abs(row['offset_s']-expected) <= 2.)
            same(row['localized_correct'], localized, 'Strata physical localization labels')
    return {'configurations': 8, 'known_queries': 543, 'metadata_defined_groups': 3,
            'all_outcomes_counts_global_gates_and_clustered_intervals_rederived': True,
            'groups': {name: value['n_queries'] for name, value in old['groups'].items()}}


def verify_replica_ranges(results, review):
    derived = []
    for n in (141, 256, 512, 659):
        for layout in ('baseline', 'compact'):
            workers = [load(results/'streaming_index'/f'{layout}_{n}_r{r}.json') for r in range(3)]
            item = {'gallery_sources': n, 'layout': layout, 'reference_audio_hours': workers[0]['reference_audio_s']/3600}
            for key, source in [('explicit_reference_MB', 'explicit_reference_bytes'),
                                ('process_peak_enrollment_rss_MB', 'process_peak_enrollment_rss_bytes'),
                                ('rss_after_enrollment_gc_MB', 'rss_after_enrollment_gc_bytes')]:
                values = [w[source]/1e6 for w in workers]
                item[key] = {'median': float(np.median(values)), 'min': min(values), 'max': max(values), 'replicates': values}
            derived.append(item)
    same(derived, load(review/'memory_replica_ranges.json')['rows'], 'All 24 streaming replica ranges')
    return {'fresh_process_replicas': 24, 'range_summary_rows': 8, 'ranges_are_confidence_intervals': False}


def verify_dense_sparse(results, review):
    dense, sparse = [load(results/'acr_medium'/f'acr_d{factor}_5s.json')['predictions'] for factor in (1, 8)]
    maps = [{r['query_id']: r for r in rows} for rows in (dense, sparse)]
    gates = [select_threshold([r for r in rows if r['role']=='calibration_unknown'], .01) for rows in (dense, sparse)]
    known = [r for r in dense if r['role']=='test_known']
    out = {}
    for metric in ('correct_reference', 'localized_correct', 'accepted_correct', 'accepted_localized'):
        paired_rows, dense_only, sparse_only = [], 0, 0
        for row in known:
            a, b = [m[row['query_id']] for m in maps]
            label = 'correct_reference' if metric=='accepted_correct' else 'localized_correct' if metric=='accepted_localized' else metric
            x, y = bool(a[label]), bool(b[label])
            if metric.startswith('accepted'):
                x = x and a['reference_id'] is not None and a['score'] >= gates[0]
                y = y and b['reference_id'] is not None and b['score'] >= gates[1]
            dense_only += x and not y
            sparse_only += y and not x
            paired_rows.append({'source_id': row['source_id'], 'source_query_id': row['source_query_id'], 'difference': int(y)-int(x)})
        out[metric] = crossed_interval(paired_rows, 'difference') | {'sparse_only': sparse_only, 'dense_only': dense_only}
    same(out, load(review/'paired_dense_sparse_review.json'), 'All four dense/sparse paired comparisons')
    return out


def verify_unknown_waveform(results, review, root):
    directory = review/'unknown_waveform'
    plan_path = directory/'plan.json'
    plan, summary, strict = [load(directory/(name+'.json')) for name in ('plan', 'summary', 'strict_validation')]
    protocol_path = results/'extended_unknown_protocol.json'
    result_path = results/'acr_ivf_extended/acr_ivf_qs4_extended_5s.json'
    gate_path = results/'acr_ivf_extended/threshold_qs4.json'
    for path, key in ((protocol_path, 'protocol_sha256'), (result_path, 'result_sha256'), (gate_path, 'gate_sha256')):
        bind(path, plan[key], root)
    same(sha(Path(__file__).with_name('audit_acr_unknown_waveform.py')), plan['runner_sha256'], 'Waveform runner')
    check_code(plan)
    plan_sha = original_sha(plan_path, root)
    same(summary['plan_sha256'], plan_sha, 'Waveform summary plan')
    same(strict['plan_sha256'], plan_sha, 'Independent waveform audit plan')
    bind(directory/'summary.json', strict['summary_sha256'], root)
    for key, expected in (('sample_rate', 8000), ('criterion_absolute_full_crop_pearson_at_least', .95),
                          ('offset_radius_s', .25), ('minimum_full_crop_samples', 40000),
                          ('constant_query_norm_threshold', 1e-9), ('frozen_clean_false_accept_count', 16),
                          ('frozen_clean_test_count', 4000)):
        same(plan[key], expected, 'Prespecified waveform criterion/scope '+key)
    require(not any(plan[key] for key in ('primary_predictions_modified', 'primary_gate_modified', 'primary_labels_modified')), 'Changed primary results during waveform audit')
    protocol, result, gate = load(protocol_path), load(result_path), load(gate_path)
    queries = {q['query_id']: q for q in protocol['queries']}
    references = {r['reference_id']: r for r in protocol['references']}
    calibration = [r for r in result['predictions'] if r['role']=='calibration_unknown']
    require(len(calibration) == 1074 and gate['uses_extended_test_outcomes'] is False, 'Expanded calibration-only gate')
    same(select_threshold(calibration, .001), gate['threshold'], 'Expanded original gate')
    same(result['frozen_threshold'], gate['threshold'], 'Frozen expanded result gate')
    clean = [r for r in result['predictions'] if r['role']=='test_unknown' and queries[r['query_id'].split('@')[0]].get('negative_population')=='additional_clean_FMA']
    require(len(clean) == 4000, 'Expanded clean unknown cohort')
    accepted = sorted((r for r in clean if r['reference_id'] is not None and r['score'] >= gate['threshold']), key=lambda r: r['query_id'])
    require(len(accepted) == 16, 'Every frozen clean acceptance required')
    same([r['query_id'] for r in accepted], [r['query_id'] for r in plan['rows']], 'Complete frozen16 planned roster')
    same([r['query_id'] for r in accepted], [r['query_id'] for r in summary['rows']], 'Complete frozen16 measured roster')
    independent = {r['query_id']: r for r in strict['rows']}
    same(set(independent), {r['query_id'] for r in accepted}, 'Independent waveform audit coverage')
    correlations, discrepancies = [], []
    statuses = Counter()
    for prediction, planned, measured in zip(accepted, plan['rows'], summary['rows'], strict=True):
        q = queries[prediction['query_id'].split('@')[0]]
        ref = references[prediction['reference_id']]
        identity = {'query_id': prediction['query_id'], 'unknown_source_id': q['source_id'],
                    'predicted_reference_id': prediction['reference_id'], 'frozen_score': prediction['score'],
                    'predicted_reference_offset_s': prediction['offset_s'], 'query_crop_start_s': q['start_s'],
                    'query_crop_duration_s': 5., 'unknown_encoded_source_sha256': q['audio_sha256'],
                    'unknown_encoded_path': q['path'], 'reference_encoded_path': ref['path']}
        for key, value in identity.items():
            same(planned[key], value, 'Frozen waveform roster '+key)
            same(measured[key], value, 'Measured waveform roster '+key)
        same(measured['reference_encoded_source_sha256'], planned['reference_encoded_source_sha256'], 'Reference encoded digest')
        for key in ('unknown_encoded_source_sha256', 'reference_encoded_source_sha256', 'query_pcm_crop_sha256', 'reference_pcm_search_window_sha256'):
            require(len(measured[key]) == 64 and all(c in '0123456789abcdef' for c in measured[key]), 'Waveform digest format')
        same(measured['query_crop_samples'], 40000, 'Waveform query frames')
        same(measured['full_crop_search_lag_count'], max(0, measured['reference_search_window_samples']-40000+1), 'Full-crop lag geometry')
        require(0 <= measured['reference_search_window_samples'] <= 44000, 'Waveform window geometry')
        status = measured['waveform_audit_status']
        statuses[status] += 1
        checked = independent[prediction['query_id']]
        same(checked['status'], status, 'Independent status')
        require(checked['independently_checked'] is True, 'Independent audit coverage flag')
        if status == 'measured':
            value = measured['absolute_waveform_correlation']
            require(math.isfinite(value) and 0 <= value <= 1+1e-10, 'Pearson range')
            same(abs(measured['signed_waveform_correlation']), value, 'Absolute Pearson')
            same(measured['strong_waveform_overlap_0p95'], value >= .95, 'Prespecified waveform flag')
            require(abs(measured['refined_lag_from_frozen_offset_s']) <= .25+1/8000 and measured['query_centered_norm'] > 1e-9, 'Waveform lag/norm support')
            require(measured['reference_search_window_samples'] >= 40000 and len(measured['matching_reference_crop_sha256']) == 64, 'Measured full-crop support')
            require(abs(checked['numpy_fft_maximum_absolute_pearson']-value) <= 1e-8 and
                    abs(checked['numpy_fft_refined_offset_s']-measured['refined_reference_offset_s']) <= 1/16000, 'Independent FFT agreement')
            discrepancy = abs(checked['direct_peak_signed_pearson']-measured['signed_waveform_correlation'])
            require(discrepancy <= 1e-8, 'Independent direct Pearson agreement')
            discrepancies.append(discrepancy)
            correlations.append(value)
        else:
            require(status == 'insufficient_full_crop_support' and measured['reference_search_window_samples'] < 40000, 'Unsupported crop retained explicitly')
            same(measured.get('strong_waveform_overlap_0p95', False), False, 'Unsupported crop cannot flag')
    same(dict(statuses), summary['status_counts'], 'Every audit status')
    same(max(correlations), summary['max_absolute_waveform_correlation'], 'Maximum retained correlation')
    same(float(np.median(correlations)), summary['median_absolute_waveform_correlation'], 'Median retained correlation')
    same(max(discrepancies), strict['maximum_direct_pearson_discrepancy'], 'Independent discrepancy summary')
    flags = sum(r.get('strong_waveform_overlap_0p95', False) for r in summary['rows'])
    same(flags, summary['strong_shared_5s_waveform_flags'], 'All waveform flags')
    require(summary['count_or_label_changes'] == 0 and summary['frozen_false_accept_count'] == 16 and summary['clean_test_count'] == 4000, 'Original16/4000 count retained')
    return {'all_frozen_clean_acceptances': 16, 'measured_full_crops': len(correlations),
            'status_counts': dict(statuses), 'shared_waveform_flags': flags,
            'maximum_absolute_correlation': max(correlations), 'source_or_label_changes': 0,
            'roster_criterion_digests_and_independent_audit_agreement_checked': True,
            'waveform_samples_redecoded_in_this_check': False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, default=ROOT/'results')
    p.add_argument('--review', type=Path, help='New review evidence directory; defaults to results/review2026')
    p.add_argument('--output', type=Path, default=ROOT/'work/v3_evidence_recomputed.json')
    a = p.parse_args()
    review = a.review or a.results/'review2026'
    manifest = ROOT/'SHA256SUMS.json'
    if manifest.exists():
        for name, expected in load(manifest).items(): same(sha(ROOT/name), expected, f'Artifact file SHA {name}')
    data = {'version': 'V3', 'audio_free': True,
            'digital_boundary': verify_boundary(a.results, review, ROOT),
            'reference_phase': verify_phases(a.results, review, ROOT),
            'distortion_strata': verify_strata(a.results, review),
            'memory_replica_ranges': verify_replica_ranges(a.results, review),
            'paired_dense_sparse': verify_dense_sparse(a.results, review),
            'unknown_waveform': verify_unknown_waveform(a.results, review, ROOT)}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(data, indent=2)+'\n')
    print('Verified 3340 stress predictions, 6680 all-phase predictions, 8 metadata-strata configurations, 24 streaming replicas and the frozen16 waveform-audit roster; no audio/inference rerun.')


if __name__ == '__main__':
    main()
