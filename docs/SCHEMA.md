# Artifact relationships

`source SHA-256 → video_id → input manifest → inference job → answer record → scoring job → score record → summary row`

- Input `manifest.json`: videos with `video_id`, dataset, external relative source path, source SHA-256, M, J and four ordered image lists. Each list item stores its relative PNG path, hash and optional source frame ID. Identical hashes may occur repeatedly.
- `plan.json`: phase, request count, frozen file hashes and `plan_sha256`. A changed plan is a distinct run. `requests.jsonl` has one job per line.
- Inference request: video/source hash, task, condition, repeat, provider/model, prompt path/hash and ordered image entries. It contains no GT. Only prompt text and image blocks go to the API.
- Answer record: request metadata, `description`, raw description text, requested/returned model, response/request IDs, token usage, cost estimate, status and raw response hash. `index.json` links all completed records and exposes expected/completed counts.
- GT reference file: `{status, source_sha256, items:[{video_id,task,reference}]}`. No inference stage opens it.
- Scoring request: links the exact candidate answer-record hash and GT-file hash; sends only task/reference/candidate plus rubric. Candidate model, condition and repeat remain local metadata. Score output: `{score:0..5, reason:string, needs_human_review:boolean}`.
- Score index: same video-linked structure as answers. `individual_scores.csv` keeps individual repeats; `summary.json` reports descriptive statistics separately by dataset/task/condition/model. Human-review flags remain visible and are not automatically resolved.
- Shared budget ledger: counts all experiments using that ledger. Pending/unknown outcomes reserve budget; completed requests use returned usage and frozen tariffs. It is operational state, not a scientific score file.

GT1 and GT2 scores are not pooled. A single coarse score can be the same for different descriptions; use the saved descriptions and reasons to inspect substantive differences. Byte-identical responses are not expected from a stochastic hosted service.
