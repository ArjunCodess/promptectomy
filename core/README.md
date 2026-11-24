# Promptectomy

Find repetitive LLM calls in your logs. Verify Python replacements on independently checked inputs before you switch.

Install with Python 3.11 or newer.

```sh
python -m pip install promptectomy
uvx promptectomy --version
```

`ingest` accepts native JSONL, OpenAI Chat Completions and Batch exports, LangSmith chat runs, and OpenTelemetry GenAI spans. Use `analyze` to find tasks and `suggest` for a proposed replacement. `review` drafts inputs for independent labeling, and `test` saves a Markdown or HTML verdict.

See the [project README](https://github.com/ArjunCodess/promptectomy#readme) for the demo and reviewer workflow.
