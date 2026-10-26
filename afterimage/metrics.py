"""Export measured images, usage and scores without API access.

Frame counts are optional, separately collected metadata, bound by source hash.
Every output directory is new; existing answers and scores remain untouched.
"""
import csv
import statistics
from collections import defaultdict
from pathlib import Path
from .common import check, fresh, read, safe, sha, validate_inputs, write


def csv_write(path, rows):
    if not rows:
        return
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def records(folder, kind):
    folder = Path(folder)
    index = read(folder / 'index.json')
    check(index['kind'] == kind, 'Wrong result kind')
    result = {}
    for item in index['items']:
        path = safe(folder, item['path'])
        check(sha(path) == item['sha256'], 'Result hash mismatch')
        record = read(path)
        job = record['request']['job_id']
        check(record['status'] == 'completed' and job not in result, 'Invalid or duplicate result')
        result[job] = (record, item['sha256'])
    return index, result


def total_input(provider, usage):
    if provider == 'anthropic':
        return usage['input_tokens'] + usage.get('cache_creation_input_tokens', 0) + usage.get('cache_read_input_tokens', 0)
    return usage['input_tokens']


def export(inputs, answers, out, scores=None, frame_counts=None):
    manifest, validation = validate_inputs(inputs)
    videos = {v['video_id']: v for v in manifest['videos']}
    metadata = {}
    if frame_counts:
        with Path(frame_counts).open(encoding='utf-8-sig', newline='') as stream:
            for row in csv.DictReader(stream):
                check(row['video_id'] not in metadata, 'Duplicate frame metadata')
                metadata[row['video_id']] = row
    counts = []
    for v in videos.values():
        m, j = v['M'], v['J']
        n = None
        if v['video_id'] in metadata:
            meta = metadata[v['video_id']]
            check(meta['source_sha256'] == v['source_sha256'], 'Frame metadata source hash mismatch')
            n = int(meta['N_original_frames'])
            check(n > 0, 'Invalid frame count')
        counts.append(dict(video_id=v['video_id'], dataset=v['dataset'], source_sha256=v['source_sha256'],
                           N_original_frames=n, M_selected_appearances=m, J_afterimages=j,
                           selected_inputs=m+2, afterimage_inputs=j+2,
                           original_to_selection_reduction=1-m/n if n else None,
                           selection_to_composite_reduction=1-j/m if m else None,
                           VLM_input_reduction=1-(j+2)/(m+2),
                           original_to_final_input_reduction=1-(j+2)/n if n else None))
    answer_index, results = records(answers, 'inference')
    score_index, grades = (None, {}) if scores is None else records(scores, 'scoring')
    grade_by_candidate = {}
    for grade, _ in grades.values():
        q = grade['request']; key = q['candidate_job_id']
        check(key in results and key not in grade_by_candidate, 'Unmatched or duplicate score')
        check(q['candidate_record_sha256'] == results[key][1], 'Score refers to a different answer')
        grade_by_candidate[key] = grade
    rows = []
    for key, (answer, _) in results.items():
        q, usage = answer['request'], answer['usage']
        check(q['video_id'] in videos, 'Answer video missing from inputs')
        v = videos[q['video_id']]
        check(q['images'] == v['conditions'][q['condition']], 'Answer input order/hash differs from manifest')
        grade = grade_by_candidate.get(key)
        row = {k: q[k] for k in ['video_id', 'dataset', 'task', 'provider', 'model', 'condition', 'repeat']}
        row.update(input_images=len(q['images']), input_tokens_total=total_input(q['provider'], usage),
                   output_tokens_billable=usage['output_tokens'], inference_cost_usd=answer['cost_usd'],
                   score=grade['score'] if grade else None,
                   judge_cost_usd=grade['cost_usd'] if grade else None,
                   needs_human_review=grade['needs_human_review'] if grade else None,
                   description=answer['description'], judge_reason=grade['reason'] if grade else None)
        rows.append(row)
    pairs = []
    groups = defaultdict(dict)
    for row in rows:
        key = tuple(row[k] for k in ['video_id', 'dataset', 'task', 'provider', 'model', 'repeat'])
        check(row['condition'] not in groups[key], 'Duplicate condition within repeat')
        groups[key][row['condition']] = row
    for key, conditions in sorted(groups.items()):
        for baseline in ['selected_sequence', 'uniform_budget']:
            if baseline not in conditions or 'afterimage' not in conditions:
                continue
            a, b = conditions[baseline], conditions['afterimage']
            row = dict(zip(['video_id', 'dataset', 'task', 'provider', 'model', 'repeat'], key))
            row.update(baseline=baseline, baseline_images=a['input_images'], afterimage_images=b['input_images'],
                       image_reduction=1-b['input_images']/a['input_images'],
                       baseline_input_tokens=a['input_tokens_total'], afterimage_input_tokens=b['input_tokens_total'],
                       input_token_reduction=1-b['input_tokens_total']/a['input_tokens_total'] if a['input_tokens_total'] else None,
                       baseline_score=a['score'], afterimage_score=b['score'],
                       score_delta=b['score']-a['score'] if a['score'] is not None and b['score'] is not None else None)
            pairs.append(row)
    aggregates = []
    for dataset in sorted({r['dataset'] for r in counts}) + ['all']:
        subset = [r for r in counts if dataset == 'all' or r['dataset'] == dataset]
        m = sum(r['M_selected_appearances'] for r in subset); j = sum(r['J_afterimages'] for r in subset)
        n = sum(r['N_original_frames'] for r in subset) if all(r['N_original_frames'] for r in subset) else None
        k = len(subset)
        aggregates.append(dict(dataset=dataset, videos=k, N=n, M=m, J=j, selected_inputs=m+2*k, afterimage_inputs=j+2*k,
                               pooled_original_to_selection_reduction=1-m/n if n else None,
                               pooled_selection_to_composite_reduction=1-j/m if m else None,
                               pooled_VLM_input_reduction=1-(j+2*k)/(m+2*k),
                               macro_VLM_input_reduction=statistics.mean(r['VLM_input_reduction'] for r in subset)))
    out = fresh(out)
    csv_write(out/'input_counts.csv', counts)
    csv_write(out/'individual_usage_scores.csv', rows)
    csv_write(out/'paired_comparisons.csv', pairs)
    write(out/'summary.json', dict(input_validation=validation, counts=aggregates,
          answer_set_complete=answer_index['complete'], scoring_set_complete=score_index['complete'] if score_index else None,
          completed_answers=len(rows), matched_scores=len(grade_by_candidate),
          input_manifest_sha256=sha(Path(inputs)/'manifest.json'), answer_index_sha256=sha(Path(answers)/'index.json'),
          frame_metadata_sha256=sha(frame_counts) if frame_counts else None,
          note='N is optional supplied source-hash-bound metadata, not a full decode audit. M includes duplicate appearances. Counts aggregate each video once. Input tokens include text/schema/images; output tokens include billable reasoning. No raw-frame API baseline, statistical significance or quality gain is implied. Missing values stay empty.'))
    return dict(answers=len(rows), scores=len(grade_by_candidate), comparisons=len(pairs), videos=len(counts), API_calls=0)
