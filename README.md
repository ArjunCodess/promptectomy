# Promptectomy

Promptectomy finds repetitive LLM calls in your logs and tests Python replacements against independently checked answers. Use it for tasks with a clear expected result, such as extracting a ticket ID or selecting a tool with fixed arguments.

## Install

Run with uvx or install from GitHub with pip. Both need Python 3.11 or newer.

```sh
uvx --from 'git+https://github.com/ArjunCodess/promptectomy.git#subdirectory=core' promptectomy --version
python -m pip install 'git+https://github.com/ArjunCodess/promptectomy.git#subdirectory=core'
```

## Try it

```sh
cd core
uv sync
uv run python examples/demo.py
uv run python examples/tau_demo.py
```

The first command runs a fixture offline. The second downloads a pinned [τ-bench retail-agent trace](https://github.com/sierra-research/tau-bench/tree/59a200c6d575d595120f1cb70fea53cef0632f6b/historical_trajectories). It checks email authentication on one qualifying turn per benchmark task and keeps the AI for open-ended replies. The published calls lack per-call token counts and timing, so those baseline figures are unavailable. Reports are under `core/.promptectomy/`. Neither demo needs a Groq key.

## Use your own logs

```sh
promptectomy ingest calls.jsonl
promptectomy analyze
promptectomy suggest task.json --out candidate.json
promptectomy review TASK --out review.jsonl
```

Native JSONL rows contain `prompt`, `output`, `model`, `input_tokens`, and `output_tokens`. Unknown token counts can be null. Supply `step` for a named task and `timestamp` as a timezone-aware ISO date. When timestamps cover the log, it estimates monthly fees from their span with a one-day minimum. Otherwise it prints its 30-day assumption. Override with `--days` and `--prices`. Explicit call costs take precedence over preset [OpenAI rates](https://openai.com/api/pricing/).

Use `ingest --format openai` for JSONL rows containing Chat Completions `request` and `response` bodies. Use `--format openai-batch --requests INPUT.jsonl` for paired Batch files, `--format otel` for OTLP JSON with GenAI message and usage attributes, or `--format langsmith` for LLM runs with one chat input batch and one generation batch. Redact with `ingest --redact REGEX` before drafting review inputs. Verification uses the stored, redacted values.

Use the [example task](core/examples/tickets.task.json) to write a description without sensitive data. Only that description goes to Groq. With a key in `.env` or `GROQ_API_KEY`, `suggest` uses GPT OSS 20B for simple tasks and 120B for hard ones. Use `--model ID` to override it or `--model rules` to force the offline rules. Rules also run when the key is absent.

## Verify a replacement

Have someone check the prompts in `review.jsonl` against an independent reference, fill in `expected` and `source`, and set `verified: true` on the checked rows. Give the reviewer your task ID. Use a public key you received separately from that reviewer.

```sh
promptectomy review-key .promptectomy/reviewer.key reviewer.pub
promptectomy attest TASK review.jsonl .promptectomy/reviewer.key review.sig
promptectomy test candidate.json review.jsonl policy.json --reviewed \
  --reviewer-key reviewer.pub --attestation review.sig --report report.html
```

Choose [the policy](core/examples/policy.json) before review. PASS requires at least 20 distinct held-out prompts from the task and a 95% Wilson lower bound of at least 85%, or the stricter policy you set. Use `--development` to iterate without getting a PASS. Keep `state.sqlite3` beside the store. It locks task IDs and prevents a changed run from reusing inputs already scored there. The signature binds the labels to the reviewer's key. Keep the holdout with the reviewer until the candidate is ready. The evidence still depends on that review.

The candidate runs as a restricted Python function in a fresh worker with a 128 MiB memory cap and a three-second timeout per input. The sandbox allows basic operations plus parts of `re` and `json`. It blocks file and network access. Inspect exported Python before using it elsewhere.

PASS exits with 0. FAIL exits with 1 and records mismatches or sandbox violations. UNVERIFIED exits with 3 and says what's missing. KEEP_AI exits with 4 and explains why the task should stay with the model. Use a `.md` or `.html` report path, and check the report for sensitive inputs before sharing.

## Development

Run `uv run pytest` and `uv run ruff check .` in `core/`. Run `pnpm install` and `pnpm dev` in `web/`. Set Vercel's root directory to `web`. The release workflow uses PyPI Trusted Publishing in the `pypi` environment.
