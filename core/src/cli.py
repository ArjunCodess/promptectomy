import io
import math
import os
import random
import re
import sqlite3
import sys
from collections.abc import Callable
from functools import wraps
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from dotenv import dotenv_values, find_dotenv
from pydantic import TypeAdapter, ValidationError
from rich.cells import cell_len
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Column, Table
from rich.text import Text

from .evidence import COPIED, authentic, digest, key_pair, sign, transaction, used_inputs
from .importers import LogFormat, import_calls
from .models import Call, Candidate, Example, Policy, Price, TaskSpec
from .storage import load_rows, parse_rows, redact, write_rows, write_text

if TYPE_CHECKING:
    from .analysis import Task

app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)
console = Console()
errors = Console(stderr=True)
STORE = Path(".promptectomy/calls.jsonl")
DAYS = Annotated[
    float | None, typer.Option(min=0.00001, help="Override the observed log window in days.")
]
PRICES = Annotated[Path | None, typer.Option(help="JSON with USD token rates per model.")]


def print_version(value: bool):
    if value:
        console.print(f"promptectomy {version('promptectomy')}")
        raise typer.Exit()


@app.callback()
def configure_console(
    show_version: Annotated[
        bool, typer.Option("--version", callback=print_version, is_eager=True)
    ] = False,
):
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(errors="backslashreplace")


def command(function: Callable):
    @wraps(function)
    def run(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except ValidationError as error:
            for item in error.errors(include_input=False):
                location = ".".join(map(str, item["loc"])) or "document"
                errors.print(
                    Text(f"Invalid {location}: {item['type']}"), style="red", soft_wrap=True
                )
        except (OSError, ValueError, OverflowError, re.error, sqlite3.Error) as error:
            errors.print(Text(str(error)), style="red", soft_wrap=True)
        raise typer.Exit(2)

    return run


def log_window(calls: list[Call], days: float | None) -> tuple[float, str]:
    if days is not None:
        if not math.isfinite(days):
            raise ValueError("--days must be finite.")
        return days, "explicit --days"
    dates = [call.timestamp for call in calls if call.timestamp is not None]
    if len(dates) == len(calls) and len(dates) > 1 and max(dates) > min(dates):
        return max(1, (max(dates) - min(dates)).total_seconds() / 86_400), (
            "timestamp span, with a one-day minimum"
        )
    return 30, f"assumed window, no complete timestamp span ({len(dates)}/{len(calls)} dated)"


def rank_tasks(calls: list[Call], prices: bytes | None, days: float, state: Path) -> list["Task"]:
    from .analysis import analyze as analyze_calls

    rates = TypeAdapter(dict[str, Price]).validate_json(prices) if prices is not None else {}
    with transaction(state) as db:
        db.execute("CREATE TABLE IF NOT EXISTS task_ids (template TEXT PRIMARY KEY, id TEXT)")
        known = dict(db.execute("SELECT template, id FROM task_ids"))
        tasks = analyze_calls(calls, rates, days, known)
        db.executemany("INSERT OR IGNORE INTO task_ids VALUES (?, ?)", known.items())
    return tasks


@app.command()
@command
def ingest(
    source: Path,
    store: Path = STORE,
    log_format: Annotated[LogFormat, typer.Option("--format")] = LogFormat.native,
    requests: Annotated[Path | None, typer.Option(help="OpenAI Batch input JSONL.")] = None,
    redact_pattern: Annotated[
        list[str] | None,
        typer.Option("--redact", help="Remove matching text from prompts and output values."),
    ] = None,
):
    """Import native, Chat Completions, Batch, or OpenTelemetry logs."""
    patterns = [re.compile(pattern) for pattern in redact_pattern or []]
    calls, skipped = import_calls(source, log_format, requests)
    if not calls:
        raise ValueError("The log has no usable calls.")
    for call in calls:
        for pattern in patterns:
            prompt = redact(call.prompt, pattern)
            output = redact(call.output, pattern)
            call.redacted |= prompt != call.prompt or output != call.output
            call.prompt, call.output = prompt, output
    if store.exists() and not patterns:
        previous = load_rows(store, Call)
        rank_tasks(previous, None, 30, store.parent / "state.sqlite3")
    write_rows(store, calls)
    console.print(Text(f"Saved {len(calls)} calls to {store}. Skipped {skipped} records."))


@app.command()
@command
def analyze(store: Path = STORE, prices: PRICES = None, days: DAYS = None):
    """Group local calls and rank likely replacements by cost and frequency."""
    calls = load_rows(store, Call)
    window, basis = log_window(calls, days)
    tasks = rank_tasks(
        calls, prices.read_bytes() if prices else None, window, store.parent / "state.sqlite3"
    )
    id_width = max((cell_len(item.id) for item in tasks), default=4)
    compact = console.width < id_width + 66
    table = Table(
        Column("Task", no_wrap=True, min_width=id_width),
        "Pattern",
        "Calls",
        "USD / 30 days",
        "Median ms",
        "Repeat agreement",
    )
    for task in tasks:
        cost = (
            f"${task.monthly_usd:.6f} ({task.priced_calls}/{len(task.calls)})"
            if task.monthly_usd is not None
            else "-"
        )
        latency = (
            f"{task.median_ms:.2f} ({task.timed_calls}/{len(task.calls)})"
            if task.median_ms is not None
            else "-"
        )
        agreement = (
            f"{task.consistent_inputs}/{task.repeated_inputs}" if task.repeated_inputs else "-"
        )
        if compact:
            console.print(Text(task.id, style="bold"), soft_wrap=True)
            console.print(
                f"  {task.kind} | {len(task.calls)} calls | USD/30d {cost} | "
                f"median ms {latency} | repeat agreement {agreement}"
            )
        else:
            table.add_row(Text(task.id), task.kind, str(len(task.calls)), cost, latency, agreement)
    if not compact:
        console.print(table)
    console.print(f"Log window: {window:g} days ({basis}). Parentheses show data coverage.")


@app.command()
@command
def review(
    task: str,
    store: Path = STORE,
    out: Path = Path(".promptectomy/review.jsonl"),
    count: Annotated[int, typer.Option(min=1)] = 100,
    seed: int = 42,
    include_used: Annotated[bool, typer.Option(help="Include previously scored prompts.")] = False,
):
    """Draft task inputs for independent review without including model outputs."""
    state = store.parent / "state.sqlite3"
    tasks = {item.id: item for item in rank_tasks(load_rows(store, Call), None, 30, state)}
    if task not in tasks:
        raise ValueError("The task is absent from the ingested log. Run analyze.")
    used = set() if include_used else used_inputs(state)
    prompts = sorted(
        {call.prompt for call in tasks[task].calls if digest(call.prompt.encode()) not in used}
    )
    if not prompts:
        raise ValueError("No fresh task inputs are available. Import newer calls.")
    reviewed = random.Random(seed).sample(prompts, min(count, len(prompts)))
    write_rows(
        out,
        [
            Example(
                input=prompt, expected=None, verified=False, source="pending independent review"
            )
            for prompt in reviewed
        ],
    )
    console.print(Text(f"Saved {len(reviewed)} prompts for independent review to {out}."))


@app.command("review-key")
@command
def review_key(secret: Path, public: Path):
    """Create an Ed25519 key pair for an independent reviewer."""
    if secret.resolve() == public.resolve() or secret.exists() or public.exists():
        raise ValueError("Use two distinct paths for new key files.")
    private, trusted = key_pair()
    write_text(secret, private + "\n")
    write_text(public, trusted + "\n")
    console.print(Text(f"Share {public}. Keep {secret} private."))


@app.command()
@command
def attest(task: str, examples: Path, secret: Path, out: Path):
    """Sign exact JSONL bytes after checking the answers independently."""
    if out.resolve() in {examples.resolve(), secret.resolve()}:
        raise ValueError("Use a separate output path for the attestation.")
    data = examples.read_bytes()
    rows = parse_rows(examples, data, Example)
    if not any(row.verified for row in rows) or any(
        COPIED.search(row.source) for row in rows if row.verified
    ):
        raise ValueError("Attest only independently checked answers with their sources.")
    write_text(out, sign(task, data, secret.read_text(encoding="ascii").strip()) + "\n")
    console.print(Text(f"Saved attestation to {out}."))


@app.command()
@command
def suggest(
    spec: Path,
    out: Path = Path(".promptectomy/candidate.json"),
    code_out: Annotated[Path | None, typer.Option(help="Export the proposed Python.")] = None,
    model: Annotated[
        str | None, typer.Option(help="Groq model ID, or rules for offline use.")
    ] = None,
):
    """Use Groq or local rules to propose a replacement or keep the AI call."""
    from groq import APIError

    from .suggest import generate, offline

    task_spec = TaskSpec.model_validate_json(spec.read_bytes())
    local = offline(task_spec)
    if out.resolve() == spec.resolve() or (
        code_out and code_out.resolve() in {out.resolve(), spec.resolve()}
    ):
        raise ValueError("Use distinct paths for the task spec, candidate, and Python export.")
    api_key = ""
    if model != "rules" and not (
        task_spec.rule == "keep-ai" or (model is None and local.suggestion.status == "keep_ai")
    ):
        api_key = os.environ.get("GROQ_API_KEY", "").strip()
        if not api_key:
            dotenv = find_dotenv(usecwd=True)
            if dotenv:
                api_key = (dotenv_values(dotenv).get("GROQ_API_KEY") or "").strip()
        if model and not api_key:
            if sys.stdin.isatty():
                api_key = typer.prompt("GROQ_API_KEY", hide_input=True).strip()
            if not api_key:
                raise ValueError("Set GROQ_API_KEY to use this Groq model.")
    try:
        candidate = generate(task_spec, api_key, model) if api_key else local
    except APIError as error:
        raise ValueError(f"Groq request failed ({type(error).__name__}).") from None
    except SyntaxError as error:
        raise ValueError(f"Groq returned invalid Python on line {error.lineno}.") from None
    write_text(out, candidate.model_dump_json(indent=2) + "\n")
    if candidate.suggestion.status == "candidate":
        if code_out:
            write_text(code_out, candidate.suggestion.code.rstrip() + "\n")
        console.print(Syntax(candidate.suggestion.code, "python", word_wrap=True))
    elif candidate.suggestion.status == "keep_ai":
        console.print("KEEP_AI", style="cyan")
    for assumption in candidate.suggestion.assumptions:
        console.print(Text(f"- {assumption}"))
    console.print(Text(f"Saved {candidate.suggestion.status} to {out}."))


@app.command()
@command
def test(
    candidate: Path,
    examples: Path,
    policy: Path,
    store: Path = STORE,
    prices: PRICES = None,
    days: DAYS = None,
    report: Path = Path(".promptectomy/report.md"),
    reviewer_key: Annotated[Path | None, typer.Option(help="Trusted reviewer public key.")] = None,
    attestation: Annotated[Path | None, typer.Option(help="Detached evidence signature.")] = None,
    reviewed: Annotated[bool, typer.Option(help="You inspected the candidate Python.")] = False,
    development: Annotated[bool, typer.Option(help="Score development inputs without a PASS.")] = (
        False
    ),
):
    """Score a candidate on independently attested, held-out task inputs."""
    from .verify import verify

    inputs = {"candidate": candidate, "examples": examples, "policy": policy, "log": store}
    for name, path in (
        ("prices", prices),
        ("reviewer_key", reviewer_key),
        ("attestation", attestation),
    ):
        if path is not None:
            inputs[name] = path
    state = store.parent / "state.sqlite3"
    if report.resolve() in {path.resolve() for path in (*inputs.values(), state)}:
        raise ValueError("The report path must be separate from the input and state files.")
    if report.suffix.lower() not in (".md", ".html"):
        raise ValueError("Use a .md or .html report path.")
    contents = {name: path.read_bytes() for name, path in inputs.items()}
    proposal = Candidate.model_validate_json(contents["candidate"])
    criteria = Policy.model_validate_json(contents["policy"])
    cases = parse_rows(examples, contents["examples"], Example)
    logs = parse_rows(store, contents["log"], Call)
    window, basis = log_window(logs, days)
    tasks = {task.id: task for task in rank_tasks(logs, contents.get("prices"), window, state)}
    if proposal.task not in tasks:
        raise ValueError("The candidate task is absent from the ingested log. Run analyze.")
    fingerprints = {name: digest(data) for name, data in contents.items()}
    attested = authentic(
        proposal.task,
        contents["examples"],
        contents.get("attestation", b"").decode("ascii").strip(),
        contents.get("reviewer_key", b"").decode("ascii").strip(),
    )
    result = verify(
        tasks[proposal.task],
        proposal,
        cases,
        criteria,
        window,
        reviewed,
        attested=attested,
        state=state,
        development=development,
    )
    result.notes.append(f"Log window: {window:g} days ({basis}).")
    write_text(
        report,
        result.report(proposal, criteria, fingerprints, html=report.suffix.lower() == ".html"),
    )
    colors = {"PASS": "green", "FAIL": "red", "UNVERIFIED": "yellow", "KEEP_AI": "cyan"}
    console.print(Text(result.verdict, style=colors[result.verdict]))
    console.print(Text(result.reason), soft_wrap=True)
    console.print(Text(f"Report: {report}"), soft_wrap=True)
    if result.verdict != "PASS":
        raise typer.Exit({"FAIL": 1, "UNVERIFIED": 3, "KEEP_AI": 4}[result.verdict])


if __name__ == "__main__":
    app()
