import hashlib
import io
import math
import os
import re
import sys
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from dotenv import dotenv_values, find_dotenv
from pydantic import TypeAdapter, ValidationError
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from .models import Call, Candidate, Example, Policy, Price, TaskSpec
from .storage import load_rows, parse_rows, redact, write_rows, write_text

if TYPE_CHECKING:
    from .analysis import Task

app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)
console = Console()
errors = Console(stderr=True)
STORE = Path(".promptectomy/calls.jsonl")
DAYS = Annotated[float, typer.Option(min=0.01, help="Number of days covered by the logs.")]
PRICES = Annotated[Path | None, typer.Option(help="JSON with USD token rates per model.")]


@app.callback()
def configure_console():
    # Preserve Unicode in files and escape glyphs the terminal cannot display.
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
                errors.print(Text(f"Invalid {location}: {item['type']}"), style="red")
        except (OSError, ValueError, re.error) as error:
            errors.print(Text(str(error)), style="red")
        raise typer.Exit(2)

    return run


def rank_tasks(calls: list[Call], prices: bytes | None, days: float) -> list["Task"]:
    from .analysis import analyze as analyze_calls

    if not math.isfinite(days):
        raise ValueError("--days must be finite.")
    rates = TypeAdapter(dict[str, Price]).validate_json(prices) if prices is not None else {}
    return analyze_calls(calls, rates, days)


@app.command()
@command
def ingest(
    source: Path,
    store: Path = STORE,
    redact_pattern: Annotated[
        list[str] | None,
        typer.Option(
            "--redact", help="Regex to remove matching text from prompts and output values."
        ),
    ] = None,
):
    """Validate a JSONL log and write a local copy for analysis."""
    patterns = [re.compile(pattern) for pattern in redact_pattern or []]
    calls = load_rows(source, Call)
    if not calls:
        raise ValueError("The log has no calls.")
    for call in calls:
        for pattern in patterns:
            call.prompt = redact(call.prompt, pattern)
            call.output = redact(call.output, pattern)
    write_rows(store, calls)
    console.print(Text(f"Saved {len(calls)} calls to {store}."))


@app.command()
@command
def analyze(store: Path = STORE, prices: PRICES = None, days: DAYS = 30):
    """Group local calls and rank likely replacements by cost and frequency."""
    table = Table("Task", "Pattern", "Calls", "USD / 30 days", "Median ms")
    for task in rank_tasks(load_rows(store, Call), prices.read_bytes() if prices else None, days):
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
        table.add_row(Text(task.id), task.kind, str(len(task.calls)), cost, latency)
    console.print(table)
    console.print(f"Log window: {days:g} days. Parentheses show data coverage.")


@app.command()
@command
def suggest(
    spec: Path,
    out: Path = Path(".promptectomy/candidate.json"),
    code_out: Annotated[Path | None, typer.Option(help="Export the proposed Python.")] = None,
):
    """Send a task description to Groq and save its proposed replacement."""
    from groq import APIError

    from .suggest import generate

    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        dotenv = find_dotenv(usecwd=True)
        if dotenv:
            api_key = (dotenv_values(dotenv).get("GROQ_API_KEY") or "").strip()
    if not api_key and sys.stdin.isatty():
        api_key = typer.prompt("GROQ_API_KEY", hide_input=True).strip()
    if not api_key:
        raise ValueError("Set GROQ_API_KEY in your environment or .env, then rerun suggest.")

    if code_out and code_out.resolve() == out.resolve():
        raise ValueError("--out and --code-out must use different files.")
    task_spec = TaskSpec.model_validate_json(spec.read_bytes())
    try:
        candidate = generate(task_spec, api_key)
    except APIError as error:
        raise ValueError(f"Groq request failed ({type(error).__name__}).") from None
    except SyntaxError as error:
        raise ValueError(f"Groq returned invalid Python on line {error.lineno}.") from None
    write_text(out, candidate.model_dump_json(indent=2) + "\n")
    if candidate.suggestion.status == "candidate":
        if code_out:
            write_text(code_out, candidate.suggestion.code.rstrip() + "\n")
        console.print(Syntax(candidate.suggestion.code, "python", word_wrap=True))
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
    days: DAYS = 30,
    report: Path = Path(".promptectomy/report.md"),
    reviewed: Annotated[bool, typer.Option(help="You reviewed and trust the Python to run.")] = (
        False
    ),
):
    """Score reviewed Python on the held-out share of verified examples."""
    from .verify import verify

    inputs = {"candidate": candidate, "examples": examples, "policy": policy, "log": store}
    if prices:
        inputs["prices"] = prices
    if report.resolve() in {path.resolve() for path in inputs.values()}:
        raise ValueError("The report path must be separate from the input files.")
    contents = {name: path.read_bytes() for name, path in inputs.items()}
    proposal = Candidate.model_validate_json(contents["candidate"])
    criteria = Policy.model_validate_json(contents["policy"])
    cases = parse_rows(examples, contents["examples"], Example)
    logs = parse_rows(store, contents["log"], Call)
    tasks = {task.id: task for task in rank_tasks(logs, contents.get("prices"), days)}
    if proposal.task not in tasks:
        raise ValueError("The candidate task is absent from the ingested log. Run analyze.")
    fingerprints = {name: hashlib.sha256(data).hexdigest() for name, data in contents.items()}
    result = verify(tasks[proposal.task], proposal, cases, criteria, days, reviewed)
    write_text(report, result.report(proposal, criteria, fingerprints))

    table = Table("Verdict", "Holdout", "Reason")
    colors = {"PASS": "green", "FAIL": "red", "UNVERIFIED": "yellow"}
    table.add_row(
        Text(result.verdict, style=colors[result.verdict]), str(result.withheld), result.reason
    )
    console.print(table)
    console.print(Text(f"Report: {report}"))
    if result.verdict != "PASS":
        raise typer.Exit(1 if result.verdict == "FAIL" else 3)


if __name__ == "__main__":
    app()
