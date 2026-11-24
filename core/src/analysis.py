import hashlib
import re
from dataclasses import dataclass
from statistics import median

import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances

from .models import Call, Price, TaskKind

_DATA_MARKER = re.compile(r"(?im)(?:^|\s)(?:input|email|text|data|payload|value)\s*:\s*(?=\S)")
_VARIABLES = re.compile(r'\{\{.+?\}\}|"(?:[^"\\]|\\.)*"|\b\d+(?:[.\-/]\d+)*\b')
_SIGNALS: dict[TaskKind, str] = {
    "classification": r"\b(classif\w*|categoriz\w*|choose (?:one|from)|route)\b",
    "extraction": r"\b(extract\w*|parse|pull out|find the)\b",
    "formatting": r"\b(format\w*|normaliz\w*|reformat\w*|standardiz\w*)\b",
    "conversion": r"\b(convert\w*|transform\w*)\b",
}


@dataclass
class Task:
    id: str
    kind: TaskKind
    calls: list[Call]
    monthly_usd: float | None
    priced_calls: int
    median_ms: float | None
    timed_calls: int


def template(prompt: str) -> str:
    marker = _DATA_MARKER.search(prompt)
    instruction = prompt[: marker.start()] if marker else prompt.partition("\n")[0]
    return " ".join(_VARIABLES.sub(" <value> ", instruction.casefold()).split())


def group_calls(calls: list[Call]) -> dict[str, list[Call]]:
    groups: dict[str, list[Call]] = {}
    templates: dict[str, list[Call]] = {}
    for call in calls:
        if call.step:
            groups.setdefault(call.step, []).append(call)
        else:
            templates.setdefault(template(call.prompt), []).append(call)

    names = sorted(templates)
    clusters = list(range(len(names)))
    if len(names) > 1:
        vectors = TfidfVectorizer(analyzer="char", ngram_range=(1, 4)).fit_transform(
            [name or "<empty>" for name in names]
        )
        clusters = AgglomerativeClustering(
            n_clusters=None, distance_threshold=0.3, metric="precomputed", linkage="complete"
        ).fit_predict(cosine_distances(vectors))

    members: dict[int, list[str]] = {}
    for cluster, name in zip(clusters, names, strict=True):
        members.setdefault(cluster, []).append(name)
    for keys in members.values():
        digest = hashlib.sha256("\n".join(keys).encode()).hexdigest()
        task_id = f"template-{digest[:12]}"
        while task_id in groups:
            task_id += "-"
        groups[task_id] = [call for key in keys for call in templates[key]]
    return groups


def analyze(calls: list[Call], prices: dict[str, Price], days: float) -> list[Task]:
    tasks = []
    rows = []
    for task_id, group in group_calls(calls).items():
        hints = (
            task_id.replace("_", " ").replace("-", " ")
            + " "
            + " ".join(template(call.prompt) for call in group[:20])
        )
        kind = next(
            (kind for kind, signal in _SIGNALS.items() if re.search(signal, hints, re.I)),
            "unknown",
        )
        costs = []
        elapsed = [call.latency_ms for call in group if call.latency_ms is not None]
        for call in group:
            if call.cost_usd is not None:
                costs.append(call.cost_usd)
            elif call.model in prices:
                price = prices[call.model]
                costs.append(
                    (
                        price.input_per_million * call.input_tokens
                        + price.output_per_million * call.output_tokens
                    )
                    / 1_000_000
                )
        monthly = pd.Series(costs, dtype=float).sum() * 30 / days if costs else None
        tasks.append(
            Task(
                task_id,
                kind,
                group,
                monthly,
                len(costs),
                median(elapsed) if elapsed else None,
                len(elapsed),
            )
        )
        rows.append(
            {"candidate": kind != "unknown", "cost": monthly, "calls": len(group), "task": task_id}
        )
    order = (
        pd.DataFrame(rows)
        .sort_values(
            ["candidate", "cost", "calls", "task"],
            ascending=[False, False, False, True],
            na_position="last",
        )
        .index
        if rows
        else []
    )
    return [tasks[index] for index in order]
