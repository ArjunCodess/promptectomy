import json
import os
import re
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

RecordType = TypeVar("RecordType", bound=BaseModel)


def load_rows(path: Path, schema: type[RecordType]) -> list[RecordType]:
    return parse_rows(path, path.read_bytes(), schema)


def parse_rows(path: Path, contents: bytes, schema: type[RecordType]) -> list[RecordType]:
    rows = []
    for line_number, line in enumerate(contents.removeprefix(b"\xef\xbb\xbf").split(b"\n"), 1):
        if not line.strip():
            continue
        try:
            rows.append(schema.model_validate_json(line))
        except ValidationError as error:
            detail = "; ".join(
                f"{'.'.join(map(str, item['loc'])) or 'row'} ({item['type']})"
                for item in error.errors(include_input=False)
            )
            raise ValueError(f"{path}, line {line_number}: {detail}") from None
    return rows


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False, newline="\n"
    ) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(content)
            stream.close()
            os.replace(temporary, path)
        finally:
            stream.close()
            temporary.unlink(missing_ok=True)


def write_rows(path: Path, rows: Sequence[BaseModel]) -> None:
    write_text(path, "".join(f"{row.model_dump_json()}\n" for row in rows))


def redact(value, pattern: re.Pattern, replacement: str = "[redacted]"):
    if isinstance(value, str):
        return pattern.sub(lambda _: replacement, value)
    if isinstance(value, list):
        return [redact(item, pattern, replacement) for item in value]
    if isinstance(value, dict):
        return {key: redact(item, pattern, replacement) for key, item in value.items()}
    return value


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False)
