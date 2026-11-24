<img src="web/app/icon.png" width="48" height="48" alt="">

# Promptectomy

**Find LLM calls you can replace with Python.** Promptectomy groups calls from
your logs and tests a replacement against independently checked answers, like
ticket IDs from a support register.

[![CI on main](https://github.com/ArjunCodess/promptectomy/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/ArjunCodess/promptectomy/actions/workflows/ci.yml?query=branch%3Amain)

## Install

Requires Python 3.11 or newer.

```sh
python -m pip install promptectomy
```

```sh
uvx promptectomy --version
```

## Try a demo

From a checkout, run either example. Neither needs a Groq key.

```sh
cd core
uv sync
uv run python examples/demo.py
uv run python examples/tau_demo.py
```

The offline fixture gets `PASS` for ticket IDs and `KEEP_AI` for support replies.
The τ-bench replay finds three tasks in 983 retail calls. Its email lookup
matches 20 of 22 holdout answers and returns `UNVERIFIED`. Both misses came from
replies with more than one address. The reply task returns `KEEP_AI`.
[τ-bench source](https://github.com/sierra-research/tau-bench/tree/59a200c6d575d595120f1cb70fea53cef0632f6b)

Reports go to `core/.promptectomy/`. The replay signs labels drawn from the
benchmark's user database and estimates fees from archived messages at bundled
`gpt-4o` rates. The archive lacks latency, tool definitions and protocol overhead.

## Run it on your logs

### 1. Find a task

```sh
promptectomy ingest calls.jsonl
promptectomy analyze
```

Use a step label when you have one. Without a label, grouping can split one task
or combine several. After upgrading to 0.3.1, run `analyze` again and re-attest
your evidence if an inferred task ID changed.

<details>
<summary>Log formats, labels and cost estimates</summary>

- **Native JSONL.** Include `prompt`, `output`, `model`, `input_tokens` and
  `output_tokens`. Unknown token counts can be `null`. Add a `step` label and
  a timezone-aware `timestamp` when available.
- **OpenAI Chat Completions.** Pass `--format openai` for JSONL with `request`
  and `response` bodies.
- **OpenAI Batch.** Pass `--format openai-batch --requests INPUT.jsonl`.
- **LangSmith.** Pass `--format langsmith` for LLM runs with one chat input
  batch and one generation batch. Set `extra.metadata.promptectomy_step` to
  label the task.
- **OpenTelemetry.** Pass `--format otel` for OTLP JSON with GenAI message and
  usage attributes. Set `promptectomy.step` to label the task.

Use `ingest --redact REGEX` to redact inputs before review. Verification uses
the stored, redacted values.

Monthly estimates assume 30 days unless the log has a complete timestamp span,
with a one-day minimum. Override with `--days` and `--prices`. Explicit costs
take precedence over bundled rates.
[OpenAI pricing](https://openai.com/api/pricing/)

</details>

### 2. Propose a replacement

Put the task ID from `analyze` in `task.json` with a description free of
sensitive data. [Example task](core/examples/tickets.task.json)

```sh
promptectomy suggest task.json --out candidate.json
```

With `GROQ_API_KEY` in the environment or `.env`, `suggest` sends the task spec
to Groq. It uses GPT OSS 20B for simple tasks and 120B for harder ones, with one
repair attempt if needed. Logs and review files stay local.

Use `--model ID` to pick a model or `--model rules` to use local rules. The
rules also run when no key is set. Set `"rule": "keep-ai"` to keep the model
for a task.

### 3. Verify it on held-out prompts

Choose a policy before review. Replace `TASK` below with the step label or
inferred task ID. [Example policy](core/examples/policy.json)

```sh
promptectomy review TASK --out review.jsonl
```

Have a reviewer fill in `expected` and `source` from an independent reference,
then set `verified` to `true` for each checked row. Keep the holdout with the
reviewer until the candidate is ready. The reviewer signs the labels and shares
their public key with you separately.

```sh
promptectomy review-key reviewer.key reviewer.pub
promptectomy attest TASK review.jsonl reviewer.key review.sig
```

Read the candidate's Python before passing `--reviewed`.

```sh
promptectomy test candidate.json review.jsonl policy.json \
  --reviewed \
  --reviewer-key reviewer.pub \
  --attestation review.sig \
  --report report.html
```

The example policy requires at least 20 distinct holdout prompts and a 95%
Wilson lower bound of at least 85% for `PASS`. Use `--development` while
iterating. Keep `state.sqlite3` beside the store to preserve task IDs and prevent
changed runs from reusing inputs already scored there.

<details>
<summary>Exit codes and sandbox limits</summary>

- `0` **PASS.** The replacement met the policy.
- `1` **FAIL.** The report lists mismatches or sandbox violations.
- `3` **UNVERIFIED.** The report explains what fell short.
- `4` **KEEP_AI.** The task should stay with the model.

Candidates run in a restricted Python worker with a three-second timeout per
input and no file or network access. Linux and Windows enforce a 128 MiB process
limit. On macOS the parent samples resident memory and stops workers over
128 MiB, so a short allocation can overshoot between samples.

Reports can be `.md` or `.html`. Check them for sensitive inputs before sharing.
Inspect exported Python before running it outside Promptectomy.

</details>

## Development

Run `uv run pytest` and `uv run ruff check .` in `core/`. Run `pnpm install`
and `pnpm dev` in `web/`. On Vercel, set the root directory to `web`.

The release workflow uses PyPI Trusted Publishing in the `pypi` environment.
