import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import median
from typing import Literal

from pydantic import Field, TypeAdapter

from .analysis import Task
from .models import Call, Candidate, Example, FiniteJson, Policy, Record
from .storage import canonical


class Output(Record):
    value: FiniteJson
    elapsed_ms: float = Field(ge=0)


class ExecutionError(Record):
    error: str


_WORKER = Path(__file__).with_name("worker.py")
_RESULT = TypeAdapter(Output | ExecutionError)


@dataclass
class Result:
    withheld: int
    development: int
    ignored: int
    conflicts: int
    task_calls: int
    monthly_usd: float | None
    priced_calls: int
    observed_days: float
    observed_median_ms: float | None
    verdict: Literal["PASS", "FAIL", "UNVERIFIED"] = "UNVERIFIED"
    reason: str = ""
    matches: int = 0
    baseline_count: int = 0
    baseline_matches: int = 0
    comparable_candidate_matches: int = 0
    baseline_median_ms: float | None = None
    candidate_median_ms: float | None = None
    mismatches: list[dict] = field(default_factory=list)

    def report(self, candidate: Candidate, policy: Policy, fingerprints: dict[str, str]) -> str:
        data = {
            "task": candidate.task,
            "suggestion_model": candidate.model,
            "assumptions": candidate.suggestion.assumptions,
            "policy": policy.model_dump(),
            "input_sha256": fingerprints,
            **asdict(self),
        }
        # Escape fence markers so prompts and code cannot close the Markdown block.
        details = json.dumps(data, indent=2, ensure_ascii=True).replace("`", r"\u0060")
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
        overall = (
            f"{self.matches}/{self.withheld}" if self.verdict != "UNVERIFIED" else "Unavailable"
        )
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

        return (
            f"# {self.verdict}\n\n{self.reason}\n\n"
            "| Metric | Original | Replacement |\n| --- | --- | --- |\n"
            f"| Monthly model fees at observed volume | {before} | {after} |\n"
            f"| Median measured latency | {latency(self.baseline_median_ms)} "
            f"| {latency(self.candidate_median_ms)} |\n"
            f"| Exact accuracy on comparable verified inputs | {baseline} | {replacement} |\n\n"
            f"Replacement accuracy on the entire holdout: {overall}. "
            "Comparable accuracy uses inputs with one consistent logged answer. "
            "Latency uses successful replacements with paired original timings where available, "
            "and excludes replacement process startup. Monthly fees exclude local compute.\n\n"
            "The input hashes and policy reproduce the split and verdict with the same files "
            "and environment. Wall-clock timings vary. Use fresh evidence after editing code "
            "in response to a failure.\n\n"
            f"```json\n{details}\n```\n"
        )


def run_candidate(code: str, prompt: str, directory: str) -> Output | ExecutionError:
    try:
        process = subprocess.run(
            [sys.executable, "-I", "-X", "utf8", str(_WORKER)],
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
        return ExecutionError(error=f"Candidate process exited with code {process.returncode}.")
    try:
        return _RESULT.validate_json(process.stdout)
    except ValueError:
        return ExecutionError(error="Candidate did not return one JSON value without printing.")


def verify(
    task: Task,
    candidate: Candidate,
    examples: list[Example],
    policy: Policy,
    days: float,
    reviewed: bool,
) -> Result:
    grouped: dict[str, set[str]] = defaultdict(set)
    ignored = 0
    for example in examples:
        if example.verified:
            grouped[example.input].add(canonical(example.expected))
        else:
            ignored += 1
    conflicts = sum(len(values) > 1 for values in grouped.values())
    eligible = [prompt for prompt, expected in grouped.items() if len(expected) == 1]
    eligible.sort(
        key=lambda prompt: hashlib.sha256(canonical([policy.seed, prompt]).encode()).digest()
    )
    cutoff = min(max(0, len(eligible) - 1), math.ceil(len(eligible) * policy.holdout_fraction))
    held_out = eligible[:cutoff]
    result = Result(
        withheld=cutoff,
        development=len(eligible) - cutoff,
        ignored=ignored,
        conflicts=conflicts,
        task_calls=len(task.calls),
        monthly_usd=task.monthly_usd,
        priced_calls=task.priced_calls,
        observed_days=days,
        observed_median_ms=task.median_ms,
    )

    if candidate.suggestion.status == "unsupported":
        result.reason = "No deterministic replacement was proposed. Review the assumptions."
        return result
    if conflicts:
        result.reason = f"Resolve conflicting verified answers for {conflicts} distinct inputs."
        return result
    if cutoff < policy.min_holdout:
        required = max(
            policy.min_holdout + 1,
            math.floor((policy.min_holdout - 1) / policy.holdout_fraction) + 1,
        )
        examples_label = "example" if cutoff == 1 else "examples"
        result.reason = (
            f"Only {cutoff} distinct verified holdout {examples_label}. "
            f"Verify {required - len(eligible)} more distinct inputs under this policy."
        )
        return result
    if not reviewed:
        result.reason = "Review the generated Python, then rerun test with --reviewed."
        return result

    observed: dict[str, list[Call]] = defaultdict(list)
    for call in task.calls:
        observed[call.prompt].append(call)
    baseline_times = []
    matching_times = []
    candidate_times = []
    with tempfile.TemporaryDirectory(prefix="promptectomy-") as directory:
        for prompt in held_out:
            expected = next(iter(grouped[prompt]))
            original = observed[prompt]
            baseline_outputs = {canonical(call.output) for call in original}
            if len(baseline_outputs) == 1:
                result.baseline_count += 1
                result.baseline_matches += expected in baseline_outputs
            timings = [call.latency_ms for call in original if call.latency_ms is not None]
            actual = run_candidate(candidate.suggestion.code, prompt, directory)
            if isinstance(actual, ExecutionError):
                result.mismatches.append(
                    {"input": prompt, "expected": json.loads(expected), "error": actual.error}
                )
                continue

            candidate_times.append(actual.elapsed_ms)
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

    result.baseline_median_ms = median(baseline_times) if baseline_times else None
    times = matching_times or candidate_times
    result.candidate_median_ms = median(times) if times else None
    if result.matches / cutoff >= policy.min_accuracy and not any(
        "error" in mismatch for mismatch in result.mismatches
    ):
        result.verdict = "PASS"
        result.reason = "The replacement met the policy on the holdout. Review it before shipping."
    else:
        result.verdict = "FAIL"
        result.reason = "The replacement missed the accuracy threshold or failed to run."
    return result
