# GPT subscription sessions and Claude API

GPT inference can run through the installed Codex CLI using an existing ChatGPT
login. This runs a local client; the model still executes on the provider's
servers. It is not an offline/local-GPU model. Claude remains on its API route.
This combination compares two systems with different harnesses, not isolated
model quality under identical API settings.

## Prepare

Install Codex from its official distribution and sign in yourself using
`codex login`. `codex login status` must report **Logged in using ChatGPT**.
Do not copy authentication files into this repository. The tested CLI version is
`0.163.0-alpha.10.2`; later versions need renewed compatibility checks.

After generating or importing the fixed inputs:

```powershell
python -m afterimage prepare-inference --inputs runs/inputs --config configs/subscription.json --repeats 3 --out runs/subscription_plan
python -m afterimage.subscription --plan runs/subscription_plan --inputs runs/inputs --out runs/subscription_answers
```

The second command is a dry run. A full matching input set produces 4,716 GPT
requests: Weizmann GT1 1,116; UT-interaction GT1 1,440; KISA GT1 1,080; KISA GT2
1,080. GT1 and GT2 remain separate. Review the printed hash, then run:

```powershell
python -m afterimage.subscription --plan runs/subscription_plan --inputs runs/inputs --out runs/subscription_answers --execute --approve-plan REVIEWED_PLAN_HASH
```

For a separate diagnostic plan, use `--ids` and `--repeats 1` when preparing it.
Pilot answers are not reused in a distinct full plan. Use `--limit 1` on the
execution command for one new request without changing or duplicating completed
requests. Run sequentially; do not start multiple runners on one output folder.

## Isolation and records

- Each request starts a new, ephemeral session; no resume, shared conversation,
  GT, previous answer, or class-bearing filename is supplied.
- Numbered image slots preserve exact local PNG bytes, order and duplicates.
  The CLI's internal image preprocessing is not claimed to match API image detail.
- User configuration, project instructions, memory, plugins, hooks, browser and
  shell tools are disabled for the trial. Any observed tool activity invalidates
  the result. Known disabled-Code-Mode/experimental-feature notices are retained.
- The interpretation prompt is unchanged. JSON output is constrained to one
  `description`. The existing 100/180-word limits are checked without rewriting
  an invalid answer or selecting outputs by agreement with GT.
- Saved records include the session ID, requested model/effort, CLI version,
  ordered input hashes, prompt/plan hashes, events, final output and usage.
  The final event stream does not report an independently verified model version.
- Complete results and raw event hashes are checked on resume. An unresolved
  previous attempt stops the runner rather than being automatically repeated.
  Quota/authentication failures stop execution; there is no API-key fallback or
  automatic purchase/reset. Create `STOP_REQUESTED` in the output directory to
  stop before the next request. Keep the local process and computer running.

The output includes `STATUS.json`, per-video records, receipts and raw events.
`index.json`, `subscription_usage.csv` and `subscription_summary.json` update
every 25 completed records and on a normal stop/finish. Individual records and
status update after each request.

## Scientific and accounting limits

The final adapter uses a named OpenAI-auth provider with HTTP/SSE, existing ChatGPT authentication and request/stream retry limits set to zero. It does not override the reserved built-in openai provider. A 230-image (110,995,305 PNG bytes) diagnostic completed through this route without observed reconnects. An earlier WebSocket trial reconnected and then produced an HTTPS result; that trace is preserved separately and excluded from the clean pilot index.

The requested model is GPT-6.1 Sol with medium reasoning. Codex contributes its
own system harness. The API's 4,096-output-token cap and image-detail setting are
not asserted to be reproduced by this CLI route. CLI usage includes that harness
and is not an image-only or equivalent API token measurement. Record subscription
cost as unknown (`null`), never estimated from API token rates or reported as
free. Subscription usage limits still apply.

Use the separate subscription usage exporter for these records; the existing
paid-API metrics command expects numeric API costs and must not be used to
silently fill unknown costs with zero. The saved descriptions can be planned for
separate GT scoring using `prepare-scoring`; this does not authorize or launch
paid scoring. Preserve their execution-mode provenance when analyzing scores.

The author retained previous GPT API pilot records and their pending costs but
disabled further GPT main-batch submission. Claude Weizmann requests are
submitted and collected independently, without waiting for GPT.

Official references: [non-interactive Codex execution](https://learn.chatgpt.com/docs/non-interactive-mode)
and [developer commands](https://learn.chatgpt.com/docs/developer-commands?surface=cli).
