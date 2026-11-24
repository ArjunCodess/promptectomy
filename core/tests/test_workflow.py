import json
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from promptectomy.analysis import analyze as rank
from promptectomy.cli import app
from promptectomy.evidence import authentic, key_pair, sign
from promptectomy.models import Call, Candidate, Example, Policy, Price, Suggestion, TaskSpec
from promptectomy.storage import load_rows, write_rows
from promptectomy.suggest import generate, offline
from promptectomy.verify import ExecutionError, run_candidate, verify, wilson_lower
from typer.testing import CliRunner

SPEC = TaskSpec(task="extract_ticket_id", description="Extract a CT- ticket ID", rule="ticket-id")
POLICY = Policy(min_accuracy=0.85, min_holdout=20, holdout_fraction=0.5)


def test_version_without_command():
    result = CliRunner().invoke(app, ["--version"])
    assert result.exit_code == 0, result.output
    assert "promptectomy 0.2.0" in result.output


@pytest.fixture
def dataset():
    cases = []
    calls = []
    for i in range(48):
        prompt = f"Extract a ticket ID.\nText: Review CT-{i:04d}."
        cases.append(
            Example(
                input=prompt,
                expected=f"CT-{i:04d}",
                verified=True,
                source="checked against the ticket register",
            )
        )
        calls.append(
            Call(
                prompt=prompt,
                output=f"CT-{i if i % 5 else i + 1:04d}",
                model="fixture",
                step=SPEC.task,
                input_tokens=320,
                output_tokens=8,
                cost_usd=0.0006,
                latency_ms=500,
            )
        )
    return calls, cases


def score(dataset, tmp_path, *, candidate=None, policy=POLICY, **options):
    calls, cases = dataset
    return verify(
        rank(calls, {}, 30)[0],
        candidate or offline(SPEC),
        cases,
        policy,
        30,
        True,
        state=tmp_path / "state.sqlite3",
        **({"attested": True} | options),
    )


def test_end_to_end_attested_html(dataset, tmp_path):
    calls, cases = dataset
    runner = CliRunner()
    log, store, evidence = (
        tmp_path / name for name in ("log.jsonl", "calls.jsonl", "labels.jsonl")
    )
    candidate, policy, report = (
        tmp_path / name for name in ("fix.json", "policy.json", "report.html")
    )
    private, public, stamp = (
        tmp_path / name for name in ("review.key", "review.pub", "labels.sig")
    )
    write_rows(log, calls)
    write_rows(evidence, cases)
    candidate.write_text(offline(SPEC).model_dump_json())
    policy.write_text(POLICY.model_dump_json())
    commands = [
        ["ingest", str(log), "--store", str(store)],
        ["analyze", "--store", str(store)],
        ["review-key", str(private), str(public)],
        ["attest", SPEC.task, str(evidence), str(private), str(stamp)],
        [
            "test",
            str(candidate),
            str(evidence),
            str(policy),
            "--store",
            str(store),
            "--reviewed",
            "--report",
            str(report),
            "--attestation",
            str(stamp),
            "--reviewer-key",
            str(public),
        ],
    ]
    for command in commands:
        result = runner.invoke(app, command)
        assert result.exit_code == 0, result.output
    assert "PASS" in result.output
    document = report.read_text(encoding="utf-8")
    assert "<h1>PASS</h1>" in document
    assert "24/24" in document
    assert "$0.028800" in document
    assert "Wilson" in document
    assert "input_sha256" in document


def test_validation_does_not_wrap_or_overwrite(tmp_path):
    folder = tmp_path / ("long-path-" * 12)
    folder.mkdir()
    log = folder / "source.jsonl"
    store = tmp_path / "store.jsonl"
    log.write_text('{"model":"private input"}\n', encoding="utf-8")
    store.write_text("untouched")
    result = CliRunner().invoke(
        app, ["ingest", str(log), "--store", str(store)], env={"COLUMNS": "50"}
    )
    assert result.exit_code == 2
    assert "line 1" in result.output
    assert "private input" not in result.output
    assert store.read_text() == "untouched"


def test_nonfinite_nested_json_rejected_at_ingest(tmp_path):
    log = tmp_path / "calls.jsonl"
    log.write_text('{"prompt":"x","output":[NaN],"model":"m","input_tokens":1,"output_tokens":1}')
    result = CliRunner().invoke(app, ["ingest", str(log), "--store", str(tmp_path / "store.jsonl")])
    assert result.exit_code == 2
    assert "output (value_error)" in result.output


@pytest.mark.parametrize(
    "payload",
    [
        "return open({path}).read()",
        "return open({path}, 'w')",
        "import socket\n    return None",
        "import os\n    return None",
        "return prompt.__class__",
    ],
)
def test_host_access_is_blocked(dataset, tmp_path, payload):
    evidence_file = tmp_path / "answers.jsonl"
    evidence_file.write_text("independently checked")
    candidate = offline(SPEC)
    candidate.suggestion.code = "def replace(prompt):\n    " + payload.format(
        path=repr(str(evidence_file))
    )
    result = score(dataset, tmp_path, candidate=candidate)
    assert result.verdict == "FAIL"
    assert "blocked before execution" in result.reason
    assert result.evaluated == 0
    assert evidence_file.read_text() == "independently checked"


def test_worker_enforces_memory_cap(tmp_path):
    result = run_candidate("def replace(prompt): return 'a' * 268435456", "", str(tmp_path))
    assert isinstance(result, ExecutionError)
    assert "memory limit" in result.error


def test_minimum_sample_and_wilson(dataset, tmp_path):
    calls, cases = dataset
    weak = Policy(min_accuracy=0.5, min_holdout=1, holdout_fraction=0.5)
    result = score((calls, cases[:2]), tmp_path, policy=weak)
    assert result.verdict == "UNVERIFIED"
    assert result.required_holdout == 20
    assert result.required_accuracy == 0.85
    assert wilson_lower(20, 20) < 0.85 < wilson_lower(24, 24)


def test_mismatched_and_copied_inputs(dataset, tmp_path):
    calls, cases = dataset
    cases[0] = cases[0].model_copy(update={"input": "invented"})
    cases[1] = cases[1].model_copy(update={"source": "copied from model output"})
    calls[2].redacted = True
    result = score((calls, cases), tmp_path)
    assert result.verdict == "UNVERIFIED"
    assert (result.unrelated, result.copied) == (1, 1)
    assert "redacted calls" in " ".join(result.notes)


def test_attestation_binds_task_and_bytes(dataset, tmp_path):
    private, public = key_pair()
    receipt = sign(SPEC.task, b"rows", private)
    assert authentic(SPEC.task, b"rows", receipt, public)
    assert not authentic(SPEC.task, b"altered rows", receipt, public)
    assert not authentic("other task", b"rows", receipt, public)
    result = score(dataset, tmp_path, attested=False)
    assert result.verdict == "UNVERIFIED"
    assert "independent reviewer" in result.reason


def test_conflict_identifies_input(dataset, tmp_path):
    calls, cases = dataset
    conflicting = cases[0].model_copy(update={"expected": "conflict"})
    result = score((calls, [*cases, conflicting]), tmp_path)
    assert result.verdict == "UNVERIFIED"
    assert result.conflicts == [conflicting.input]


def test_exact_replay_and_changed_candidate_holdout_rejection(dataset, tmp_path, monkeypatch):
    from promptectomy.verify import Output

    monkeypatch.setattr(
        "promptectomy.verify.run_candidate",
        lambda code, prompt, directory: Output(value=prompt[-8:-1], elapsed_ms=1),
    )
    assert score(dataset, tmp_path).verdict == "PASS"
    assert score(dataset, tmp_path).verdict == "PASS"
    changed = offline(SPEC)
    changed.suggestion.code += "\n# new candidate\n"
    result = score(dataset, tmp_path, candidate=changed)
    assert result.verdict == "UNVERIFIED"
    assert result.reused == 24


def test_development_can_iterate_but_cannot_pass(dataset, tmp_path, monkeypatch):
    from promptectomy.verify import Output

    monkeypatch.setattr(
        "promptectomy.verify.run_candidate",
        lambda code, prompt, directory: Output(value=prompt[-8:-1], elapsed_ms=1),
    )
    result = score(dataset, tmp_path, development=True)
    assert result.verdict == "UNVERIFIED"
    assert result.matches == 24
    assert result.holdout_sha256 is None


def test_failed_exact_accuracy(dataset, tmp_path, monkeypatch):
    from promptectomy.verify import Output

    monkeypatch.setattr(
        "promptectomy.verify.run_candidate",
        lambda code, prompt, directory: Output(value=None, elapsed_ms=1),
    )
    result = score(dataset, tmp_path)
    assert result.verdict == "FAIL"
    assert len(result.mismatches) == 24


def test_good_point_score_with_weak_confidence_is_unverified(dataset, tmp_path, monkeypatch):
    from promptectomy.verify import Output

    outcomes = iter([False, False] + [True] * 22)

    def run(code, prompt, directory):
        return Output(value=prompt[-8:-1] if next(outcomes) else None, elapsed_ms=1)

    monkeypatch.setattr("promptectomy.verify.run_candidate", run)
    result = score(dataset, tmp_path)
    assert result.matches == 22
    assert result.verdict == "UNVERIFIED"
    assert "confidence bound" in result.reason


def test_rules_without_key_and_keep_ai(tmp_path, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr("promptectomy.cli.find_dotenv", lambda **options: "")
    runner = CliRunner()
    spec, out = tmp_path / "task.json", tmp_path / "candidate.json"
    for description, status in [(SPEC.description, "candidate"), ("Draft a reply", "keep_ai")]:
        spec.write_text(TaskSpec(task=SPEC.task, description=description).model_dump_json())
        result = runner.invoke(app, ["suggest", str(spec), "--out", str(out)])
        assert result.exit_code == 0, result.output
        assert Candidate.model_validate_json(out.read_bytes()).suggestion.status == status


def test_task_ids_and_prices_persist(dataset, tmp_path):
    calls, _ = dataset
    first = calls[1].model_copy(update={"step": None, "cost_usd": None, "model": "gpt-4o-mini"})
    variant = first.model_copy(update={"prompt": "Extract the ticket ID.\nInput: CT-0073"})
    registry = {}
    before = rank([first], {}, 30, registry)[0]
    after = rank([variant, first, first], {}, 30, registry)[0]
    assert after.id == before.id
    assert len(after.calls) == 3
    assert (after.consistent_inputs, after.repeated_inputs) == (1, 1)
    assert after.monthly_usd == pytest.approx(3 * (320 * 0.15 + 8 * 0.6) / 1_000_000)
    override = Price(input_per_million=1, output_per_million=2)
    assert rank([first], {"gpt-4o-mini": override}, 15)[0].monthly_usd == pytest.approx(0.000672)


def test_batch_import_and_review_draft(tmp_path):
    runner = CliRunner()
    requests, result_file, store, draft = (
        tmp_path / name
        for name in ("requests.jsonl", "results.jsonl", "calls.jsonl", "draft.jsonl")
    )
    requested = {
        "custom_id": "native-job",
        "body": {
            "model": "gpt-4o-mini",
            "step": "extraction",
            "messages": [{"role": "user", "content": "Extract CT-5120"}],
        },
    }
    received = {
        "custom_id": "native-job",
        "response": {
            "status_code": 200,
            "body": {
                "created": 1730000000,
                "choices": [{"message": {"content": "CT-5120"}}],
                "usage": {"prompt_tokens": 8, "completion_tokens": 3},
            },
        },
    }
    requests.write_text(json.dumps(requested))
    result_file.write_text(json.dumps(received))
    result = runner.invoke(
        app,
        [
            "ingest",
            str(result_file),
            "--format",
            "openai-batch",
            "--requests",
            str(requests),
            "--store",
            str(store),
        ],
    )
    assert result.exit_code == 0, result.output
    assert load_rows(store, Call)[0].timestamp == datetime.fromtimestamp(1730000000, UTC)
    result = runner.invoke(
        app, ["review", "extraction", "--store", str(store), "--out", str(draft)]
    )
    assert result.exit_code == 0, result.output
    case = load_rows(draft, Example)[0]
    assert case.input == "Extract CT-5120"
    assert not case.verified and case.expected is None


def test_refusal_is_not_unverified(dataset, tmp_path):
    candidate = Candidate(
        task=SPEC.task,
        model="rules",
        suggestion=Suggestion(
            status="keep_ai", code="", assumptions=["Keep the model for context-based judgment."]
        ),
    )
    result = score(dataset, tmp_path, candidate=candidate, attested=False)
    assert result.verdict == "KEEP_AI"


def test_open_telemetry_import_uses_usage_and_timestamps(tmp_path):
    attributes = [
        {"key": key, "value": {field: value}}
        for key, field, value in [
            ("gen_ai.operation.name", "stringValue", "chat"),
            ("gen_ai.response.model", "stringValue", "gpt-4o-mini"),
            ("gen_ai.input.messages", "stringValue", "Extract the ID"),
            ("gen_ai.output.messages", "stringValue", "CT-0010"),
            ("gen_ai.usage.input_tokens", "intValue", "80"),
            ("gen_ai.usage.output_tokens", "intValue", "6"),
        ]
    ]
    span = {
        "name": "extract",
        "startTimeUnixNano": "1730000000000000000",
        "endTimeUnixNano": "1730000000250000000",
        "attributes": attributes,
    }
    log, store = tmp_path / "otel.json", tmp_path / "calls.jsonl"
    log.write_text(json.dumps({"resourceSpans": [{"scopeSpans": [{"spans": [span]}]}]}))
    result = CliRunner().invoke(
        app, ["ingest", str(log), "--format", "otel", "--store", str(store)]
    )
    assert result.exit_code == 0, result.output
    call = load_rows(store, Call)[0]
    assert (call.prompt, call.output, call.input_tokens) == ("Extract the ID", "CT-0010", 80)
    assert call.latency_ms == 250


def test_langsmith_import_and_long_ids(tmp_path):
    log, store = tmp_path / "langsmith.jsonl", tmp_path / "calls.jsonl"
    step = "call_a_real_model_with_a_long_step_label"
    run = {
        "name": step,
        "run_type": "llm",
        "start_time": "2026-01-01T00:00:00Z",
        "end_time": "2026-01-01T00:00:00.400Z",
        "inputs": {"messages": [[{"type": "human", "content": "Extract the ID"}]]},
        "outputs": {
            "generations": [[{"text": "CT-0010"}]],
            "llm_output": {"model_name": "gpt-4o-mini"},
        },
    }
    later = {**run, "start_time": "2026-01-08T00:00:00Z", "end_time": "2026-01-08T00:00:00.400Z"}
    log.write_text(json.dumps(run) + "\n" + json.dumps(later))
    runner = CliRunner()
    result = runner.invoke(
        app, ["ingest", str(log), "--format", "langsmith", "--store", str(store)]
    )
    assert result.exit_code == 0, result.output
    assert load_rows(store, Call)[0].latency_ms == 400
    result = runner.invoke(app, ["analyze", "--store", str(store)], env={"COLUMNS": "40"})
    assert result.exit_code == 0, result.output
    assert step in result.output
    assert "Log window: 7 days" in result.output


@pytest.mark.parametrize(
    "complexity, override, expected",
    [
        ("simple", None, "openai/gpt-oss-20b"),
        ("hard", None, "openai/gpt-oss-120b"),
        ("hard", "openai/gpt-oss-20b", "openai/gpt-oss-20b"),
    ],
)
def test_suggest_model_and_payload(monkeypatch, complexity, override, expected):
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
    assert generate(task, "fake-key", override).model == expected
    assert len(requests) == 1
    assert requests[0]["messages"][1] == {"role": "user", "content": "Return null."}
    assert "private label" not in repr(requests)
