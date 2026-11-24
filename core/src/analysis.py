import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass
from statistics import median

import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances

from .models import Call, Price, TaskKind
from .storage import canonical

_DATA_MARKER = re.compile(r"(?im)(?:^|\s)(?:input|email|text|data|payload|value)\s*:\s*(?=\S)")
_VARIABLES = re.compile(r'\{\{.+?\}\}|"(?:[^"\\]|\\.)*"|\b\d+(?:[.\-/]\d+)*\b')
_SIGNALS: dict[TaskKind, str] = {
    "classification": r"\b(classif\w*|categoriz\w*|choose (?:one|from)|route)\b",
    "extraction": r"\b(extract\w*|parse|pull out|find the)\b",
    "formatting": r"\b(format\w*|normaliz\w*|reformat\w*|standardiz\w*)\b",
    "conversion": r"\b(convert\w*|transform\w*)\b",
}
DEFAULT_PRICES = {
    "gpt-4o-mini": Price(input_per_million=0.15, output_per_million=0.60),
    "gpt-4o-mini-2024-07-18": Price(input_per_million=0.15, output_per_million=0.60),
    "gpt-4o": Price(input_per_million=2.50, output_per_million=10.00),
    "gpt-4o-2024-08-06": Price(input_per_million=2.50, output_per_million=10.00),
    "gpt-4o-2024-11-20": Price(input_per_million=2.50, output_per_million=10.00),
    "gpt-4.1-mini": Price(input_per_million=0.40, output_per_million=1.60),
    "gpt-4.1-mini-2025-04-14": Price(input_per_million=0.40, output_per_million=1.60),
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
    repeated_inputs: int = 0
    consistent_inputs: int = 0


def template(prompt: str) -> str:
    marker = _DATA_MARKER.search(prompt)
    instruction = prompt[: marker.start()] if marker else prompt.partition("\n")[0]
    return " ".join(_VARIABLES.sub(" <value> ", instruction.casefold()).split())


def group_calls(calls: list[Call], known: dict[str, str]) -> dict[str, list[Call]]:
    groups: dict[str, list[Call]] = {}
    templates: dict[str, list[Call]] = {}
    for call in calls:
        if call.step:
            groups.setdefault(call.step, []).append(call)
        else:
            templates.setdefault(template(call.prompt), []).append(call)

    if not templates:
        return groups
    if set(known.values()) & groups.keys():
        raise ValueError("A step label collides with a persisted task ID. Rename the step.")
    names = sorted(templates.keys() | known.keys())
    positions = {name: index for index, name in enumerate(names)}
    clusters = list(range(len(names)))
    distances = None
    if len(names) > 1:
        vectors = TfidfVectorizer(analyzer="char", ngram_range=(1, 4)).fit_transform(
            [name or "<empty>" for name in names]
        )
        distances = cosine_distances(vectors)
        clusters = AgglomerativeClustering(
            n_clusters=None, distance_threshold=0.3, metric="precomputed", linkage="complete"
        ).fit_predict(distances)

    members: dict[int, list[str]] = {}
    for cluster, name in zip(clusters, names, strict=True):
        members.setdefault(cluster, []).append(name)
    taken = set(known.values()) | groups.keys()
    for keys in members.values():
        anchors = [key for key in keys if key in known]
        new_keys = [key for key in keys if key not in known]
        if not anchors:
            digest = hashlib.sha256("\n".join(keys).encode()).hexdigest()
            task_id = f"template-{digest[:12]}"
            while task_id in taken:
                task_id += "-"
            taken.add(task_id)
            known.update(dict.fromkeys(new_keys, task_id))
            continue
        if new_keys:
            assert distances is not None
            for key in new_keys:
                nearest = min(anchors, key=lambda item: distances[positions[key], positions[item]])
                known[key] = known[nearest]
    for key, items in templates.items():
        groups.setdefault(known[key], []).extend(items)
    return groups


def analyze(
    calls: list[Call],
    prices: dict[str, Price],
    days: float,
    known: dict[str, str] | None = None,
) -> list[Task]:
    tasks = []
    rows = []
    rates = DEFAULT_PRICES | prices
    for task_id, group in group_calls(calls, known if known is not None else {}).items():
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
        outputs = defaultdict(list)
        for call in group:
            outputs[call.prompt].append(canonical(call.output))
            if call.cost_usd is not None:
                costs.append(call.cost_usd)
            elif (
                call.model in rates
                and call.input_tokens is not None
                and call.output_tokens is not None
            ):
                price = rates[call.model]
                costs.append(
                    (
                        price.input_per_million * call.input_tokens
                        + price.output_per_million * call.output_tokens
                    )
                    / 1_000_000
                )
        monthly = float(pd.Series(costs, dtype=float).sum()) * 30 / days if costs else None
        repeated = [values for values in outputs.values() if len(values) > 1]
        tasks.append(
            Task(
                task_id,
                kind,
                group,
                monthly,
                len(costs),
                median(elapsed) if elapsed else None,
                len(elapsed),
                len(repeated),
                sum(len(set(values)) == 1 for values in repeated),
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
