import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from promptectomy.analysis import analyze
from promptectomy.cli import app
from promptectomy.models import Call, Candidate, Example, Policy, Price, TaskSpec
from promptectomy.storage import load_rows
from promptectomy.suggest import generate
from promptectomy.verify import verify
from typer.testing import CliRunner

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.fixture(scope="module")
def data():
    calls = load_rows(EXAMPLES / "calls.jsonl", Call)
    tasks = {task.id: task for task in analyze(calls, {}, 30)}
    return calls, tasks


def test_ingest_analyze_and_report(tmp_path):
    runner = CliRunner()
    store = tmp_path / "calls.jsonl"
    report = tmp_path / "report.md"
    args = ["--store", str(store)]
    ingested = runner.invoke(
        app, ["ingest", str(EXAMPLES / "calls.jsonl"), *args, "--redact", "needs review"]
    )
    assert ingested.exit_code == 0, ingested.output
    assert "[redacted]" in store.read_text()
    ranked = runner.invoke(app, ["analyze", *args])
    assert ranked.exit_code == 0
    assert "extract_ticket_id" in ranked.output

    ingested = runner.invoke(app, ["ingest", str(EXAMPLES / "calls.jsonl"), *args])
    assert ingested.exit_code == 0
    result = runner.invoke(
        app,
        [
            "test",
            str(EXAMPLES / "tickets.candidate.json"),
            str(EXAMPLES / "tickets.verified.jsonl"),
            str(EXAMPLES / "policy.json"),
            *args,
            "--report",
            str(report),
            "--reviewed",
        ],
    )
    assert result.exit_code == 0, result.output
    content = report.read_text(encoding="utf-8")
    assert "# PASS" in content
    assert "4/4" in content
    assert "2/4 | 4/4" in content
    assert "$0.004800" in content
    assert "input_sha256" in content


def test_bad_log_does_not_overwrite_existing_store(tmp_path):
    runner = CliRunner()
    log = tmp_path / "source.jsonl"
    store = tmp_path / "store.jsonl"
    log.write_text('{"model":"private input"}\n', encoding="utf-8")
    store.write_text("untouched")
    result = runner.invoke(app, ["ingest", str(log), "--store", str(store)])
    assert result.exit_code == 2
    assert "line 1" in result.output
    assert "private input" not in result.output
    assert store.read_text() == "untouched"


def test_nonfinite_nested_json_rejected_at_ingest(tmp_path):
    runner = CliRunner()
    log = tmp_path / "calls.jsonl"
    log.write_text(
        '{"prompt":"x","output":[NaN],"model":"model","input_tokens":1,"output_tokens":1}\n',
        encoding="utf-8",
    )
    store = tmp_path / "store.jsonl"
    result = runner.invoke(app, ["ingest", str(log), "--store", str(store)])
    assert result.exit_code == 2
    assert "output (value_error)" in result.output
    assert not store.exists()


def test_labels_templates_and_rates(data):
    calls, _ = data
    chosen = [call.model_copy(update={"step": None, "cost_usd": None}) for call in calls]
    chosen.insert(1, chosen[0].model_copy(update={"prompt": "Extract ticket IDs.\nInput: CT-7348"}))
    tasks = analyze(chosen, {"fixture-model": Price(input_per_million=1, output_per_million=2)}, 15)
    tickets, support = tasks
    assert tickets.kind == "extraction"
    assert len(tickets.calls) == 9
    assert tickets.monthly_usd == pytest.approx(9 * (320 + 16) * 2 / 1_000_000)
    assert support.kind == "classification"


@pytest.mark.parametrize(
    "case, reviewed, verdict",
    [
        ("tickets", False, "UNVERIFIED"),
        ("support", True, "UNVERIFIED"),
        ("tickets", True, "FAIL"),
    ],
)
def test_policy(data, case, reviewed, verdict):
    _, tasks = data
    candidate = Candidate.model_validate_json((EXAMPLES / f"{case}.candidate.json").read_bytes())
    examples = load_rows(EXAMPLES / f"{case}.verified.jsonl", Example)
    policy = Policy.model_validate_json((EXAMPLES / "policy.json").read_bytes())
    if verdict == "FAIL":
        candidate.suggestion.code = (
            "import re\n"
            "def replace(prompt):\n"
            "    match = re.search(r'CT-[0-9]{4}', prompt)\n"
            "    return match.group() if match else None\n"
        )
        examples += [examples[0], examples[0].model_copy(update={"verified": False})]
    result = verify(tasks[candidate.task], candidate, examples, policy, 30, reviewed)
    assert result.verdict == verdict
    if case == "support":
        assert "Verify 5 more distinct inputs" in result.reason
    if verdict == "FAIL":
        assert result.withheld == 4
        assert result.matches == 3
        assert result.ignored == 1
        assert result.mismatches[0]["expected"] is None
        assert result.mismatches[0]["actual"] == "CT-2105"


def test_conflicting_answers_block_execution(data):
    _, tasks = data
    candidate = Candidate.model_validate_json((EXAMPLES / "tickets.candidate.json").read_bytes())
    examples = load_rows(EXAMPLES / "tickets.verified.jsonl", Example)
    examples.append(examples[0].model_copy(update={"expected": "conflict"}))
    result = verify(
        tasks[candidate.task], candidate, examples, Policy(min_accuracy=1, min_holdout=4), 30, True
    )
    assert result.verdict == "UNVERIFIED"
    assert result.conflicts == 1


@pytest.mark.parametrize(
    "complexity, model",
    [
        ("simple", "openai/gpt-oss-20b"),
        ("hard", "openai/gpt-oss-120b"),
    ],
)
def test_suggest_sends_only_the_description(monkeypatch, complexity, model):
    requests = []

    def create(**kwargs):
        requests.append(kwargs)
        content = json.dumps(
            {"status": "candidate", "code": "def replace(prompt): return None", "assumptions": []}
        )
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    class Client:
        def __init__(self, **kwargs):
            assert kwargs["max_retries"] == 0

        def __enter__(self):
            return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

        def __exit__(self, *exc_info):
            pass

    monkeypatch.setattr("promptectomy.suggest.Groq", Client)
    task = TaskSpec(task="private label", description="Return null.", complexity=complexity)
    assert generate(task, "fake-key").model == model
    assert len(requests) == 1
    assert requests[0]["messages"][1] == {"role": "user", "content": "Return null."}
    assert "private label" not in repr(requests)
