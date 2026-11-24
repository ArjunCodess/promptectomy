import json
import math
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from html import escape
from pathlib import Path
from statistics import median
from typing import Literal

from pydantic import Field, TypeAdapter

from .analysis import Task
from .evidence import COPIED, MIN_ACCURACY, MIN_HOLDOUT, Z_95, claim_holdout, digest
from .models import Call, Candidate, Example, FiniteJson, Policy, Record
from .storage import canonical
from .worker import MAX_INPUT, validate_code


class Output(Record):
    value: FiniteJson
    elapsed_ms: float = Field(ge=0)


class ExecutionError(Record):
    error: str
    isolation_unavailable: bool = False


_WORKER = Path(__file__).with_name("worker.py")
_RESULT = TypeAdapter(Output | ExecutionError)


@dataclass
class Result:
    task_calls: int
    monthly_usd: float | None
    priced_calls: int
    observed_days: float
    observed_median_ms: float | None
    required_holdout: int
    required_accuracy: float
    verdict: Literal["PASS", "FAIL", "UNVERIFIED", "KEEP_AI"] = "UNVERIFIED"
    reason: str = ""
    withheld: int = 0
    development: int = 0
    evaluated: int = 0
    ignored: int = 0
    unrelated: int = 0
    copied: int = 0
    conflicts: list[str] = field(default_factory=list)
    reused: int = 0
    holdout_sha256: str | None = None
    lower_bound: float | None = None
    matches: int = 0
    baseline_count: int = 0
    baseline_matches: int = 0
    comparable_candidate_matches: int = 0
    baseline_median_ms: float | None = None
    candidate_median_ms: float | None = None
    mismatches: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def report(
        self, candidate: Candidate, policy: Policy, fingerprints: dict[str, str], *, html: bool
    ) -> str:
        data = {
            "task": candidate.task,
            "suggestion_model": candidate.model,
            "assumptions": candidate.suggestion.assumptions,
            "policy": policy.model_dump(),
            "input_sha256": fingerprints,
            **asdict(self),
        }
        details = json.dumps(data, indent=2, ensure_ascii=True)
        baseline = (
            f"{self.baseline_matches}/{self.baseline_count}"
            if self.baseline_count
            else "Unavailable"
        )
        replacement = (
            f"{self.comparable_candidate_matches}/{self.baseline_count}"
            if self.baseline_count
            else "Unavailable"
        )
        scores = f"{self.matches}/{self.evaluated}" if self.evaluated else "Unavailable"
        bound = f"{self.lower_bound:.1%}" if self.lower_bound is not None else "Unavailable"
        before = (
            f"${self.monthly_usd:.6f} ({self.priced_calls}/{self.task_calls} calls priced)"
            if self.monthly_usd is not None
            else "Unavailable"
        )
        after = (
            "$0 in model fees if the replacement handles every call"
            if self.verdict == "PASS"
            else "Unavailable"
        )

        def latency(value: float | None) -> str:
            return f"{value:.4f} ms" if value is not None else "Unavailable"

        rows = [
            ("Monthly model fees at observed volume", before, after),
            (
                "Median measured latency",
                latency(self.baseline_median_ms),
                latency(self.candidate_median_ms),
            ),
            ("Exact accuracy on comparable verified inputs", baseline, replacement),
        ]
        summary = (
            f"Exact matches: {scores}. Lower endpoint of the 95% Wilson interval: {bound}. "
            f"Required holdout: {self.required_holdout}. Required accuracy: "
            f"{self.required_accuracy:.1%}."
        )
        methods = (
            "Comparable accuracy uses prompts with one consistent logged answer. Latency uses "
            "successful replacements with paired original timings where available, excluding "
            "process startup. Monthly model fees exclude local compute."
        )
        if html:
            cells = "".join(
                f"<tr><th scope='row'>{escape(label)}</th><td>{escape(before)}</td>"
                f"<td>{escape(after)}</td></tr>"
                for label, before, after in rows
            )
            notes = "".join(f"<p>{escape(note)}</p>" for note in self.notes)
            return (
                "<!doctype html><html lang='en'><meta charset='utf-8'>"
                f"<title>Promptectomy | {self.verdict}</title><main><h1>{self.verdict}</h1>"
                f"<p>{escape(self.reason)}</p><p>{escape(summary)}</p><table>"
                "<thead><tr><th>Metric</th><th>Original</th><th>Replacement</th></tr></thead>"
                f"<tbody>{cells}</tbody></table><p>{escape(methods)}</p>{notes}"
                f"<h2>Evidence and mismatches</h2><pre>{escape(details)}</pre></main></html>\n"
            )
        rows_md = "".join(f"| {label} | {before} | {after} |\n" for label, before, after in rows)
        notes_md = "".join(f"\n{note}\n" for note in self.notes)
        details = details.replace("`", r"\u0060")
        return (
            f"# {self.verdict}\n\n{self.reason}\n\n{summary}\n\n"
            "| Metric | Original | Replacement |\n| --- | --- | --- |\n"
            f"{rows_md}\n{methods}\n{notes_md}\n"
            f"```json\n{details}\n```\n"
        )


def run_candidate(code: str, prompt: str, directory: str) -> Output | ExecutionError:
    if len(prompt.encode()) > MAX_INPUT:
        return ExecutionError(error="Input exceeds the 1 MiB sandbox limit.")
    try:
        process = subprocess.run(
            [sys.executable, "-I", "-S", "-X", "utf8", str(_WORKER)],
            input=canonical({"code": code, "input": prompt}),
            capture_output=True,
            encoding="utf-8",
            timeout=3,
            cwd=directory,
            env={key: os.environ[key] for key in ("SystemRoot", "WINDIR") if key in os.environ},
        )
    except subprocess.TimeoutExpired:
        return ExecutionError(error="Candidate exceeded the 3-second limit.")
    if process.returncode:
        return ExecutionError(error=f"Candidate worker exited with code {process.returncode}.")
    try:
        return _RESULT.validate_json(process.stdout)
    except ValueError:
        return ExecutionError(error="Candidate worker did not return a valid result.")


def wilson_lower(matches: int, count: int) -> float:
    rate = matches / count
    square = Z_95**2
    center = rate + square / (2 * count)
    margin = Z_95 * math.sqrt((rate * (1 - rate) + square / (4 * count)) / count)
    return max(0, (center - margin) / (1 + square / count))


def verify(
    task: Task,
    candidate: Candidate,
    examples: list[Example],
    policy: Policy,
    days: float,
    reviewed: bool,
    *,
    attested: bool,
    state: Path,
    development: bool = False,
) -> Result:
    result = Result(
        task_calls=len(task.calls),
        monthly_usd=task.monthly_usd,
        priced_calls=task.priced_calls,
        observed_days=days,
        observed_median_ms=task.median_ms,
        required_holdout=max(MIN_HOLDOUT, policy.min_holdout),
        required_accuracy=max(MIN_ACCURACY, policy.min_accuracy),
    )
    if candidate.suggestion.status == "keep_ai":
        result.verdict = "KEEP_AI"
        result.reason = "Keep the AI call. See the refusal reasons and next steps in the report."
        return result
    if candidate.suggestion.status == "unsupported":
        result.reason = "No replacement available. Choose a matching offline rule or use Groq."
        return result

    code = candidate.suggestion.code
    try:
        validate_code(code)
    except (ValueError, SyntaxError, RecursionError) as error:
        result.verdict = "FAIL"
        result.reason = f"Candidate blocked before execution: {error}"
        return result

    observed: dict[str, list[Call]] = defaultdict(list)
    for call in task.calls:
        observed[call.prompt].append(call)
    if any(call.redacted for call in task.calls):
        result.notes.append(
            "This task has redacted calls. Evidence must use the stored prompts and the original "
            "comparison uses stored outputs. Create a new review draft from the redacted log."
        )

    grouped: dict[str, set[str]] = defaultdict(set)
    for example in examples:
        if not example.verified:
            result.ignored += 1
        elif example.input not in observed:
            result.unrelated += 1
        elif COPIED.search(example.source):
            result.copied += 1
        else:
            grouped[example.input].add(canonical(example.expected))
    if result.unrelated or result.copied:
        result.reason = (
            f"{result.unrelated} verified rows do not come from this task and "
            f"{result.copied} cite model output or copying. Correct the evidence and attest again."
        )
        return result
    result.conflicts = sorted(prompt for prompt, answers in grouped.items() if len(answers) > 1)
    if result.conflicts:
        result.reason = f"Resolve {len(result.conflicts)} conflicting inputs listed in the report."
        return result
    if not attested:
        result.reason = "Evidence needs a valid attestation from a trusted independent reviewer."
        return result

    eligible = sorted(grouped, key=lambda prompt: digest(canonical([policy.seed, prompt]).encode()))
    cutoff = min(max(0, len(eligible) - 1), math.ceil(len(eligible) * policy.holdout_fraction))
    result.withheld = cutoff
    result.development = len(eligible) - cutoff
    if not development:
        if result.required_accuracy == 1:
            result.reason = "A 100% confidence threshold cannot be met with a finite sample."
            return result
        needed = max(
            result.required_holdout,
            math.ceil(Z_95**2 * result.required_accuracy / (1 - result.required_accuracy)),
        )
        if cutoff < needed:
            total = max(needed + 1, math.floor((needed - 1) / policy.holdout_fraction) + 1)
            result.reason = (
                f"{cutoff} distinct held-out inputs available. Verify {total - len(eligible)} "
                f"more distinct task inputs for at least {needed} holdout examples."
            )
            return result
    if not reviewed:
        result.reason = "Inspect the candidate, then rerun with --reviewed."
        return result

    selected = eligible[cutoff:] if development else eligible[:cutoff]
    if not selected:
        result.reason = "No independently verified development examples are available."
        return result
    pairs = [(prompt, next(iter(grouped[prompt]))) for prompt in selected]
    result.holdout_sha256 = digest(canonical(pairs).encode()) if not development else None
    run = (
        "development"
        if development
        else digest(
            canonical(
                [task.id, digest(code.encode()), result.holdout_sha256, policy.model_dump()]
            ).encode()
        )
    )
    result.reused = claim_holdout(state, run, selected)
    if result.reused:
        result.reason = (
            f"{result.reused} sample inputs were already used in a different verification run. "
            "Use fresh evidence for this run."
        )
        return result

    baseline_times = []
    matching_times = []
    candidate_times = []
    all_match_logged_output = True
    with tempfile.TemporaryDirectory(prefix="promptectomy-") as directory:
        for prompt, expected in pairs:
            original = observed[prompt]
            baseline_outputs = {canonical(call.output) for call in original}
            all_match_logged_output &= expected in baseline_outputs
            actual = run_candidate(code, prompt, directory)
            if isinstance(actual, ExecutionError) and actual.isolation_unavailable:
                result.reason = actual.error
                return result
            result.evaluated += 1
            if len(baseline_outputs) == 1:
                result.baseline_count += 1
                result.baseline_matches += expected in baseline_outputs
            if isinstance(actual, ExecutionError):
                result.mismatches.append(
                    {"input": prompt, "expected": json.loads(expected), "error": actual.error}
                )
                continue
            candidate_times.append(actual.elapsed_ms)
            timings = [call.latency_ms for call in original if call.latency_ms is not None]
            if timings:
                baseline_times.append(median(timings))
                matching_times.append(actual.elapsed_ms)
            if canonical(actual.value) == expected:
                result.matches += 1
                result.comparable_candidate_matches += len(baseline_outputs) == 1
            else:
                result.mismatches.append(
                    {"input": prompt, "expected": json.loads(expected), "actual": actual.value}
                )
    if all_match_logged_output:
        result.notes.append(
            "Every scored expected answer also appeared in the task's model output. "
            "The independent reviewer attested to these answers."
        )

    result.baseline_median_ms = median(baseline_times) if baseline_times else None
    times = matching_times or candidate_times
    result.candidate_median_ms = median(times) if times else None
    result.lower_bound = wilson_lower(result.matches, result.evaluated)
    if any("error" in item for item in result.mismatches):
        result.verdict = "FAIL"
        result.reason = "The replacement failed to execute on at least one scored input."
    elif development:
        result.reason = "Development score only. Use fresh held-out evidence for a verdict."
    elif result.matches / result.evaluated < result.required_accuracy:
        result.verdict = "FAIL"
        result.reason = "The replacement missed the exact accuracy threshold."
    elif result.lower_bound < result.required_accuracy:
        result.reason = "The confidence bound is below the policy. Collect fresh evidence."
    else:
        result.verdict = "PASS"
        result.reason = "The replacement met the policy on independently verified holdout inputs."
    return result
