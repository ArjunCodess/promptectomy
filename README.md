# Promptectomy

If your agent calls a model to pull a ticket ID out of a string, you can probably replace that call with Python. Promptectomy finds tasks like this in your JSONL logs, estimates what they cost, and tests replacements against answers you've checked independently.

## Try it

Run this from the repo root with Python 3.11 or newer and uv. [Install uv](https://docs.astral.sh/uv/)

```sh
cd core
uv sync
uv run python examples/demo.py
```

The demo writes two reports to `.promptectomy/demo/`. Ticket extraction passes. Support routing needs more checked examples, so it comes back UNVERIFIED. The demo uses fixture data and runs without a Groq key.

## Use your own logs

Write one JSON object per line with `prompt`, `output`, `model`, `input_tokens`, and `output_tokens`. Add `step` for stable task names and `latency_ms` and `cost_usd` when available. Pass `--days` to `analyze` and `test` with the log's actual window, or they'll assume 30 days. `--prices` accepts per-model `input_per_million` and `output_per_million` rates in USD. Use `ingest --redact REGEX` to mask matching text in prompts and output values before writing the local copy.

Pick a task from `analyze` and copy [the example task](core/examples/tickets.task.json). Put a generic description in it and run `uv run promptectomy suggest task.json --out candidate.json`. Only that description goes to Groq. Set `GROQ_API_KEY` in your environment or copy `.env.example` to `.env` at the repo root. `suggest` prompts for a missing key in an interactive terminal. `complexity` selects GPT OSS 20B for `simple` or 120B for `hard`.

For `test`, supply JSONL rows with the exact `input` prompt, `expected` JSON answer, `verified: true`, and a `source` describing how you checked it. Set [your policy](core/examples/policy.json) up front. Inspect the Python before running `uv run promptectomy test candidate.json verified.jsonl policy.json --reviewed`. That flag runs the candidate as your local user. PASS exits with code 0 and is ready for review. FAIL exits with 1 and lists mismatches. UNVERIFIED exits with 3 and tells you what's missing. Check reports for sensitive inputs before sharing.

## Development

Run `uv run pytest` and `uv run ruff check .` in `core/`. For the landing page, run `pnpm install` and `pnpm dev` in `web/`. Set the Vercel project's root directory to `web`. The release workflow uses PyPI Trusted Publishing for `.github/workflows/publish.yml` in the `pypi` environment.
