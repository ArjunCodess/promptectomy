import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEMO = Path(".promptectomy/demo")
STORE = ["--store", str(DEMO / "calls.jsonl")]

commands = [
    (["ingest", "examples/calls.jsonl", *STORE], 0),
    (["analyze", *STORE], 0),
    *[
        (
            [
                "test",
                f"examples/{task}.candidate.json",
                f"examples/{task}.verified.jsonl",
                "examples/policy.json",
                *STORE,
                "--reviewed",
                "--report",
                str(DEMO / f"{task}.md"),
            ],
            expected,
        )
        for task, expected in (("tickets", 0), ("support", 3))
    ],
]

for args, expected in commands:
    result = subprocess.run([sys.executable, "-m", "promptectomy.cli", *args], cwd=ROOT)
    if result.returncode != expected:
        raise SystemExit(f"Expected exit {expected} from {args[0]}, got {result.returncode}.")
