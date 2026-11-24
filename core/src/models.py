import math
from typing import Annotated, Literal

from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
)

Label = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Amount = Annotated[float, Field(ge=0)]
TaskKind = Literal["extraction", "formatting", "conversion", "classification", "unknown"]


def require_finite(value: JsonValue) -> JsonValue:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("JSON numbers must be finite.")
    if isinstance(value, list):
        for item in value:
            require_finite(item)
    if isinstance(value, dict):
        for item in value.values():
            require_finite(item)
    return value


FiniteJson = Annotated[JsonValue, AfterValidator(require_finite)]


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class Call(Record):
    prompt: str
    output: FiniteJson
    model: Label
    input_tokens: Annotated[int, Field(ge=0)] | None
    output_tokens: Annotated[int, Field(ge=0)] | None
    step: Label | None = None
    cost_usd: Amount | None = None
    latency_ms: Amount | None = None
    timestamp: AwareDatetime | None = None
    redacted: bool = False


class Price(Record):
    input_per_million: Amount
    output_per_million: Amount


class TaskSpec(Record):
    task: Label
    description: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=12000)
    ]
    complexity: Literal["simple", "hard"] = "simple"
    rule: Literal["ticket-id", "iso-date", "email", "trim", "lowercase", "keep-ai"] | None = None


class Suggestion(Record):
    status: Literal["candidate", "unsupported", "keep_ai"]
    code: str
    assumptions: list[Label]


class Candidate(Record):
    task: Label
    model: Label
    suggestion: Suggestion


class Example(Record):
    input: str
    expected: FiniteJson
    verified: bool
    source: Label


class Policy(Record):
    min_accuracy: Annotated[float, Field(gt=0, le=1)]
    min_holdout: Annotated[int, Field(ge=1)]
    holdout_fraction: Annotated[float, Field(gt=0, lt=1)] = 0.25
    seed: int = 42
