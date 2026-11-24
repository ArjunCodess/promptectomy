import hashlib
import re
import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from .storage import canonical

MIN_HOLDOUT = 20
MIN_ACCURACY = 0.85
Z_95 = 1.959963984540054
COPIED = re.compile(r"\b(copied|from (?:the )?(?:ai|model|llm)|(?:model|ai|llm) output)\b", re.I)
EVIDENCE_USE = "CREATE TABLE IF NOT EXISTS evidence_use (input_id TEXT PRIMARY KEY, run_id TEXT)"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def key_pair() -> tuple[str, str]:
    private = Ed25519PrivateKey.generate()
    return private.private_bytes_raw().hex(), private.public_key().public_bytes_raw().hex()


def sign(task: str, data: bytes, private: str) -> str:
    message = canonical(["promptectomy-evidence-v1", task, digest(data)]).encode()
    return Ed25519PrivateKey.from_private_bytes(bytes.fromhex(private)).sign(message).hex()


def authentic(task: str, data: bytes, receipt: str, public: str) -> bool:
    message = canonical(["promptectomy-evidence-v1", task, digest(data)]).encode()
    try:
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(public)).verify(
            bytes.fromhex(receipt), message
        )
    except (ValueError, InvalidSignature):
        return False
    return True


@contextmanager
def transaction(state: Path) -> Iterator[sqlite3.Connection]:
    state.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(state, timeout=30)) as db, db:
        db.execute("BEGIN IMMEDIATE")
        yield db


def claim_holdout(state: Path, run: str, inputs: list[str]) -> int:
    with transaction(state) as db:
        db.execute(EVIDENCE_USE)
        ids = [(digest(prompt.encode()),) for prompt in inputs]
        conflicts = sum(
            db.execute(
                "SELECT 1 FROM evidence_use WHERE input_id = ? AND run_id <> ?", (*item, run)
            ).fetchone()
            is not None
            for item in ids
        )
        if not conflicts:
            db.executemany(
                "INSERT OR IGNORE INTO evidence_use VALUES (?, ?)",
                [(*item, run) for item in ids],
            )
        return conflicts


def used_inputs(state: Path) -> set[str]:
    with transaction(state) as db:
        db.execute(EVIDENCE_USE)
        return {row[0] for row in db.execute("SELECT input_id FROM evidence_use")}
