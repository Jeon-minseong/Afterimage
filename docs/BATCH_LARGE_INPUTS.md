# Large inputs and staged submission

The default batch part size remains 24,000,000 bytes. Claude's Message Batches endpoint permits 256 MB per request; this runner allows an explicitly selected maximum of 240,000,000 bytes to leave framing overhead. The OpenAI path retains its previous 24 MB limit.

```powershell
python -m afterimage prepare-batch --plan runs/claude_plan --inputs data/inputs --out runs/claude_part_a --provider anthropic --max-bytes 240000000 --job-ids JOB_ID_1 JOB_ID_2
```

`--job-ids` selects complete existing requests from the frozen plan. It does not alter image order, duplicate appearances, prompts, or repetitions. Omit it to prepare all requests for that provider. Multiple disjoint packs can share one plan and answer directory. Their answer index reports progress against the entire plan. The shared ledger rejects overlapping submission of a job from the same plan.

A single interpretation request is never split across model calls. PNG files are sent byte-for-byte; the runner does not resize, recompress, grid, or drop images. Token counting uses smaller calls for oversized inputs and sums their counts with repeated prompt/schema overhead. Such counts are reservation estimates, not measured inference usage; `count_metadata` records the distinction in each pack. Actual response usage determines the recorded cost.

Submit only parts whose maximum output and input reservation, plus prior spending and all outstanding reservations, fit the cumulative provider cap. Collect finished batches to replace reservations with measured costs before admitting more work. An expected low output cost does not justify reserving beyond the cap. Unknown submissions must be reconciled, never automatically retried.

The October 26 continuation excludes the completed 1,116 Weizmann Claude responses and contains 1,440 UT GT1, 1,080 KISA GT1, and 1,080 KISA GT2 requests. The model, structured output, prompts, four conditions, and three repeats match the completed Weizmann stage. A largest-input KISA request from the main plan is checked before the rest of KISA is submitted; it is counted once as a main response, not run again as an extra repetition. This is a transport check, not evidence of interpretation accuracy.

Limits checked against the official [API overview](https://platform.claude.com/docs/en/api/overview), [batch guide](https://platform.claude.com/docs/en/build-with-claude/batch-processing), and [vision guide](https://platform.claude.com/docs/en/build-with-claude/vision). These limits can change; validate them before a new reproduction.
