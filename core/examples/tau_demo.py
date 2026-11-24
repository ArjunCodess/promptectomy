"""Score an unlabelled agent trace against the published tau-bench retail user records."""

import argparse
import json
import sys
from pathlib import Path
from urllib.request import urlopen

import tiktoken
from demo import DEMO, ROOT, run
from promptectomy.cli import rank_tasks
from promptectomy.evidence import digest, key_pair, sign
from promptectomy.models import Call, Candidate, Example, Policy, Suggestion, TaskSpec
from promptectomy.storage import canonical, load_rows, write_rows, write_text

REVISION = "59a200c6d575d595120f1cb70fea53cef0632f6b"
BASE_URL = f"https://raw.githubusercontent.com/sierra-research/tau-bench/{REVISION}"
TRACE = "historical_trajectories/gpt-4o-retail.json"
USERS = "tau_bench/envs/retail/data/users.json"
CHECKSUMS = {
    TRACE: "df01707894836168ff0ec9616b0bf08f66c7e5afcf313e5fe4f7a2f5c2ec938b",
    USERS: "6f4fd104cc5e80b92166c4035315c31532c6cea52a90dcdf18425ee7f87c2b82",
}
TOOL = "find_user_id_by_email"
CACHE = DEMO.parent / "tau-demo"
WORK = CACHE / "unlabelled"
CODE = r"""
import re
import json

def replace(prompt):
    messages = json.loads(prompt)
    matches = re.findall(r'[A-Za-z0-9_.+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}', messages[-1]["content"])
    if len(matches) != 1:
        return None
    return {"name": "find_user_id_by_email", "arguments": {"email": matches[0]}}
""".strip()


def load_upstream(source: str, cached: Path, local: Path | None):
    if local is None and not cached.exists():
        with urlopen(f"{BASE_URL}/{source}", timeout=60) as response:
            raw = response.read(12 * 1024 * 1024)
    else:
        raw = (local or cached).read_bytes()
    if digest(raw) != CHECKSUMS[source]:
        raise ValueError(f"Checksum does not match the pinned upstream file: {source}")
    if local is None and not cached.exists():
        write_text(cached, raw.decode())
    return json.loads(raw)


def prepare(trajectories: list[dict], users: dict) -> tuple[list[dict], list[Example]]:
    encoding = tiktoken.encoding_for_model("gpt-4o")
    calls = []
    evidence = []
    seen: set[int] = set()
    for row in trajectories:
        history = []
        for turn in row["traj"]:
            role = turn["role"]
            tool_calls = turn.get("tool_calls") or []
            if role == "assistant" and history and history[-1]["role"] == "user":
                prompt = canonical(history)
                known_email = users[row["info"]["task"]["user_id"]]["email"]
                asked_for_email = (
                    len(history) >= 3
                    and history[-2]["role"] == "assistant"
                    and "email" in (history[-2].get("content") or "").lower()
                )
                # The benchmark supplies the account identity. The observed reply and
                # replacement play no part in selecting or labeling a reference turn.
                if (
                    asked_for_email
                    and known_email.casefold() in (history[-1]["content"] or "").casefold()
                    and row["task_id"] not in seen
                ):
                    seen.add(row["task_id"])
                    evidence.append(
                        Example(
                            input=prompt,
                            expected={"name": TOOL, "arguments": {"email": known_email}},
                            verified=True,
                            source=f"tau-bench retail users.json, task {row['task_id']}",
                        )
                    )
                observed = [
                    {
                        "name": item["function"]["name"],
                        "arguments": json.loads(item["function"]["arguments"]),
                    }
                    for item in tool_calls
                ]
                output = observed[0] if len(observed) == 1 else observed or turn.get("content")
                calls.append(
                    {
                        "request": {"model": "gpt-4o", "messages": history.copy()},
                        "response": {
                            "choices": [{"message": {"content": output}}],
                            "usage": {
                                "prompt_tokens": len(encoding.encode_ordinary(prompt)),
                                "completion_tokens": len(
                                    encoding.encode_ordinary(canonical(output))
                                ),
                            },
                        },
                        "estimated_tokens": True,
                    }
                )
            if tool_calls or role == "tool":
                break
            history.append(turn)
    return calls, evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, help="Local copy of the pinned retail trace.")
    parser.add_argument("--users", type=Path, help="Local copy of the pinned retail user file.")
    options = parser.parse_args()
    traces = load_upstream(TRACE, ROOT / CACHE / "trace.json", options.trace)
    users = load_upstream(USERS, ROOT / CACHE / "users.json", options.users)
    records, evidence = prepare(traces, users)
    exchange_path = WORK / "openai.jsonl"
    write_text(ROOT / exchange_path, "".join(canonical(row) + "\n" for row in records))
    store_path = WORK / "calls.jsonl"
    store = ["--store", str(store_path)]
    run(["ingest", str(exchange_path), "--format", "openai", *store])
    ranked = rank_tasks(load_rows(ROOT / store_path, Call), None, 30, ROOT / WORK / "state.sqlite3")
    references = {row.input: row for row in evidence}
    auth = max(
        ranked, key=lambda group: len({call.prompt for call in group.calls} & references.keys())
    )
    print(f"Discovered {len(ranked)} tasks from {len(records)} unlabelled calls.", flush=True)
    print(
        f"Email lookup group {auth.id}: {len(auth.calls)} calls, "
        f"${auth.monthly_usd:.4f} estimated model fees for the logged volume.",
        flush=True,
    )
    labels = WORK / "examples.jsonl"
    auth_inputs = {call.prompt for call in auth.calls}
    selected = [row for prompt, row in references.items() if prompt in auth_inputs]
    if len(selected) < 23:
        raise SystemExit(f"Found only {len(selected)} reference turns in the discovered auth task.")
    write_rows(ROOT / labels, selected)
    policy = WORK / "policy.json"
    write_text(
        ROOT / policy,
        Policy(
            min_accuracy=0.85,
            min_holdout=20,
            holdout_fraction=0.96,
        ).model_dump_json(),
    )
    private, public = key_pair()
    write_text(ROOT / WORK / "benchmark-demo.pub", public + "\n")
    receipt = sign(auth.id, (ROOT / labels).read_bytes(), private)
    write_text(ROOT / WORK / "auth.sig", receipt + "\n")
    candidate = Candidate(
        task=auth.id,
        model="handwritten",
        suggestion=Suggestion(
            status="candidate",
            code=CODE,
            assumptions=[
                "Uses the supplied email after an authentication request.",
                "The benchmark user database supplies reference emails. The demo signs the labels.",
                "Token estimates omit unlogged tool schemas and protocol overhead.",
            ],
        ),
    )
    write_text(ROOT / WORK / "auth.json", candidate.model_dump_json(indent=2))
    run(
        [
            "test",
            str(WORK / "auth.json"),
            str(labels),
            str(policy),
            *store,
            "--reviewed",
            "--reviewer-key",
            str(WORK / "benchmark-demo.pub"),
            "--attestation",
            str(WORK / "auth.sig"),
            "--report",
            str(WORK / "auth.html"),
        ],
        3,
    )
    reply = max(
        (
            group
            for group in ranked
            if group.id != auth.id
            and group.repeated_inputs > group.consistent_inputs
            and all(isinstance(call.output, str) for call in group.calls)
        ),
        key=lambda group: group.repeated_inputs - group.consistent_inputs,
    )
    spec = TaskSpec(
        task=reply.id,
        rule="keep-ai",
        description=(
            f"The agent drafts retail replies using conversation context. "
            f"Identical inputs varied on {reply.repeated_inputs - reply.consistent_inputs} of "
            f"{reply.repeated_inputs} repeated prompts. Review authentication separately."
        ),
    )
    write_text(ROOT / WORK / "reply.task.json", spec.model_dump_json(indent=2))
    empty = WORK / "reply.verified.jsonl"
    write_text(ROOT / empty, "")
    run(
        [
            "suggest",
            str(WORK / "reply.task.json"),
            "--model",
            "rules",
            "--out",
            str(WORK / "reply.json"),
        ]
    )
    run(
        [
            "test",
            str(WORK / "reply.json"),
            str(empty),
            str(policy),
            *store,
            "--report",
            str(WORK / "reply.html"),
        ],
        4,
    )
    print(f"Published reference files: {BASE_URL}/{TRACE} and {BASE_URL}/{USERS}", file=sys.stderr)


if __name__ == "__main__":
    main()
