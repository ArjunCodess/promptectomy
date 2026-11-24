"""Run the bundled fixture without an API key."""

import subprocess
import sys
from pathlib import Path

from promptectomy.evidence import key_pair, sign
from promptectomy.storage import write_text

ROOT = Path(__file__).resolve().parents[1]
DEMO = Path(".promptectomy/demo")
STORE = ["--store", str(DEMO / "calls.jsonl")]


def run(args: list[str], expected: int = 0):
    result = subprocess.run([sys.executable, "-m", "promptectomy.cli", *args], cwd=ROOT)
    if result.returncode != expected:
        raise SystemExit(f"Expected exit {expected} from {args[0]}, got {result.returncode}.")


def main():
    evidence = "examples/tickets.verified.jsonl"
    private, public = key_pair()
    write_text(ROOT / DEMO / "fixture.pub", public + "\n")
    signature = sign("extract_ticket_id", (ROOT / evidence).read_bytes(), private)
    write_text(ROOT / DEMO / "tickets.sig", signature + "\n")
    run(["ingest", "examples/calls.jsonl", *STORE])
    run(["analyze", *STORE])
    for task, expected in (("tickets", 0), ("support", 4)):
        candidate = str(DEMO / f"{task}.candidate.json")
        run(["suggest", f"examples/{task}.task.json", "--model", "rules", "--out", candidate])
        proof = (
            [
                "--reviewer-key",
                str(DEMO / "fixture.pub"),
                "--attestation",
                str(DEMO / "tickets.sig"),
            ]
            if task == "tickets"
            else []
        )
        run(
            [
                "test",
                candidate,
                f"examples/{task}.verified.jsonl",
                "examples/policy.json",
                *STORE,
                "--reviewed",
                "--report",
                str(DEMO / f"{task}.html"),
                *proof,
            ],
            expected,
        )


if __name__ == "__main__":
    main()
