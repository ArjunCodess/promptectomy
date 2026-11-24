"""Score an existing agent using the published tau-bench retail trajectories."""

import argparse
import json
import string
import sys
from pathlib import Path
from urllib.request import urlopen

from demo import DEMO, ROOT, run
from promptectomy.evidence import digest, key_pair, sign
from promptectomy.models import Candidate, Example, Suggestion, TaskSpec
from promptectomy.storage import canonical, write_rows, write_text

REVISION = "59a200c6d575d595120f1cb70fea53cef0632f6b"
TRACE = (
    f"https://raw.githubusercontent.com/sierra-research/tau-bench/{REVISION}"
    "/historical_trajectories/gpt-4o-retail.json"
)
SHA256 = "df01707894836168ff0ec9616b0bf08f66c7e5afcf313e5fe4f7a2f5c2ec938b"
AUTH = "retail_email_auth"
REPLY = "retail_reply"
TOOL = "find_user_id_by_email"
ADDRESS_CHARS = string.ascii_letters + string.digits
WORK = DEMO.parent / "tau-demo"
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


def load_trace(path: Path | None) -> list[dict]:
    cached = ROOT / WORK / "trace.json"
    if path is None and not cached.exists():
        with urlopen(TRACE, timeout=60) as response:
            raw = response.read(12 * 1024 * 1024)
    else:
        raw = (path or cached).read_bytes()
    if digest(raw) != SHA256:
        raise ValueError("The retail trace checksum does not match the pinned upstream file.")
    if path is None and not cached.exists():
        write_text(cached, raw.decode())
    return json.loads(raw)


def exchange(step: str, messages: list, output: object) -> dict:
    return {
        "step": step,
        "request": {"model": "gpt-4o", "messages": messages.copy()},
        "response": {"choices": [{"message": {"content": output}}]},
    }


def supplied_emails(text: str) -> list[str]:
    """Read email tokens for the reference labels, independently of the candidate regex."""
    emails = []
    for word in text.split():
        token = word.strip(".,;:!?()<>[]{}\"'\x60*")
        local, at, domain = token.partition("@")
        hostname, dot, suffix = domain.rpartition(".")
        if (
            at
            and dot
            and local
            and hostname
            and len(suffix) >= 2
            and suffix.isascii()
            and suffix.isalpha()
            and all(character in ADDRESS_CHARS + "_.+-" for character in local)
            and all(character in ADDRESS_CHARS + ".-" for character in hostname)
        ):
            emails.append(token)
    return emails


def prepare(trajectories: list[dict]) -> tuple[list[dict], list[Example]]:
    calls = []
    evidence = []
    seen: set[int] = set()
    for row in trajectories:
        history = []
        for turn in row["traj"]:
            role = turn["role"]
            content = turn.get("content")
            tool_calls = turn.get("tool_calls") or []
            if role == "assistant" and history and history[-1]["role"] == "user":
                # Select inputs from the pre-auth conversation before looking at the response.
                asked_for_email = (
                    len(history) >= 3
                    and history[-2]["role"] == "assistant"
                    and "email" in (history[-2].get("content") or "").lower()
                )
                emails = supplied_emails(history[-1]["content"] or "") if asked_for_email else []
                if len(emails) == 1 and row["task_id"] not in seen:
                    seen.add(row["task_id"])
                    expected = {"name": TOOL, "arguments": {"email": emails[0]}}
                    evidence.append(
                        Example(
                            input=canonical(history),
                            expected=expected,
                            verified=True,
                            source="tau-bench retail auth policy and supplied user address",
                        )
                    )
                    observed = [
                        {
                            "name": item["function"]["name"],
                            "arguments": json.loads(item["function"]["arguments"]),
                        }
                        for item in tool_calls
                    ]
                    output = observed[0] if len(observed) == 1 else observed or content
                    calls.append(exchange(AUTH, history, output))
                elif content and not tool_calls:
                    calls.append(exchange(REPLY, history, content))
            if tool_calls or role == "tool":
                break
            history.append(turn)
    return calls, evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, help="Local copy of the retail trace.")
    options = parser.parse_args()
    records, evidence = prepare(load_trace(options.trace))
    exchange_path = WORK / "openai.jsonl"
    labels = WORK / "examples.jsonl"
    write_text(ROOT / exchange_path, "".join(canonical(row) + "\n" for row in records))
    write_rows(ROOT / labels, evidence)
    private, public = key_pair()
    write_text(ROOT / WORK / "upstream-demo.pub", public + "\n")
    receipt = sign(AUTH, (ROOT / labels).read_bytes(), private)
    write_text(ROOT / WORK / "auth.sig", receipt + "\n")
    candidate = Candidate(
        task=AUTH,
        model="handwritten",
        suggestion=Suggestion(
            status="candidate",
            code=CODE,
            assumptions=[
                "An unauthenticated user supplies one email after being asked for it.",
                "Compare tool name and arguments. Transport call IDs are omitted.",
            ],
        ),
    )
    write_text(ROOT / WORK / "auth.json", candidate.model_dump_json(indent=2))
    spec = TaskSpec(
        task=REPLY, description="Draft a reply to a retail request using conversation context."
    )
    write_text(ROOT / WORK / "reply.task.json", spec.model_dump_json(indent=2))
    empty = WORK / "reply.verified.jsonl"
    write_text(ROOT / empty, "")
    store = ["--store", str(WORK / "calls.jsonl")]
    run(["ingest", str(exchange_path), "--format", "openai", *store])
    run(["analyze", *store])
    run(
        [
            "test",
            str(WORK / "auth.json"),
            str(labels),
            "examples/policy.json",
            *store,
            "--reviewed",
            "--reviewer-key",
            str(WORK / "upstream-demo.pub"),
            "--attestation",
            str(WORK / "auth.sig"),
            "--report",
            str(WORK / "auth.html"),
        ]
    )
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
            "examples/policy.json",
            *store,
            "--report",
            str(WORK / "reply.html"),
        ],
        4,
    )
    print(f"Published trajectory source: {TRACE}", file=sys.stderr)


if __name__ == "__main__":
    main()
