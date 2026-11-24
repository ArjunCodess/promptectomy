from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path

from pydantic import AwareDatetime, BaseModel, Field, JsonValue

from .models import Call
from .storage import canonical, load_rows


class LogFormat(StrEnum):
    native = "native"
    openai = "openai"
    openai_batch = "openai-batch"
    otel = "otel"
    langsmith = "langsmith"


class Request(BaseModel):
    messages: list[dict[str, JsonValue]] = Field(min_length=1)
    model: str
    step: str | None = None


class Usage(BaseModel):
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)


class Choice(BaseModel):
    message: dict[str, JsonValue]


class Response(BaseModel):
    model: str | None = None
    created: int | None = None
    choices: list[Choice] = Field(min_length=1)
    usage: Usage | None = None


class Exchange(BaseModel):
    request: Request
    response: Response
    step: str | None = None
    latency_ms: float | None = None
    cost_usd: float | None = None

    def call(self) -> Call:
        messages = self.request.messages
        first = messages[0].get("content")
        prompt = (
            first
            if len(messages) == 1 and messages[0].get("role") == "user" and isinstance(first, str)
            else canonical(messages)
        )
        outputs = [
            choice.message.get("content")
            if not any(
                choice.message.get(key) for key in ("tool_calls", "function_call", "refusal")
            )
            else choice.message
            for choice in self.response.choices
        ]
        return Call(
            prompt=prompt,
            output=outputs[0] if len(outputs) == 1 else outputs,
            model=self.response.model or self.request.model,
            step=self.step or self.request.step,
            input_tokens=self.response.usage.prompt_tokens if self.response.usage else None,
            output_tokens=self.response.usage.completion_tokens if self.response.usage else None,
            latency_ms=self.latency_ms,
            cost_usd=self.cost_usd,
            timestamp=(
                datetime.fromtimestamp(self.response.created, UTC)
                if self.response.created is not None
                else None
            ),
        )


class BatchRequest(BaseModel):
    custom_id: str
    body: Request


class BatchResponse(BaseModel):
    status_code: int
    body: dict[str, JsonValue]


class BatchResult(BaseModel):
    custom_id: str
    response: BatchResponse | None = None


class SmithInputs(BaseModel):
    messages: list[list[dict[str, JsonValue]]] = Field(min_length=1, max_length=1)


class SmithGeneration(BaseModel):
    text: JsonValue
    message: dict[str, JsonValue] | None = None

    def output(self) -> JsonValue:
        if self.message is None:
            return self.text
        fields = self.message.get("kwargs", self.message)
        if isinstance(fields, dict) and (fields.get("tool_calls") or fields.get("function_call")):
            return fields
        return self.text


class SmithModelOutput(BaseModel):
    model_name: str | None = None
    token_usage: Usage | None = None


class SmithOutputs(BaseModel):
    generations: list[list[SmithGeneration]] = Field(min_length=1, max_length=1)
    llm_output: SmithModelOutput | None = None


class SmithRun(BaseModel):
    name: str
    run_type: str
    inputs: dict[str, JsonValue]
    outputs: dict[str, JsonValue] | None = None
    start_time: AwareDatetime | None = None
    end_time: AwareDatetime | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    def call(self) -> Call | None:
        if self.run_type != "llm" or self.outputs is None:
            return None
        messages = SmithInputs.model_validate(self.inputs).messages[0]
        outputs = SmithOutputs.model_validate(self.outputs)
        metadata = outputs.llm_output
        texts = [item.output() for item in outputs.generations[0]]
        usage = metadata.token_usage if metadata else None
        if not messages or not texts:
            raise ValueError("LangSmith LLM runs need messages and at least one generation.")
        elapsed = (
            (self.end_time - self.start_time).total_seconds() * 1_000
            if self.end_time and self.start_time
            else None
        )
        return Call(
            prompt=canonical(messages),
            output=texts[0] if len(texts) == 1 else texts,
            model=(metadata.model_name if metadata else None) or self.name,
            input_tokens=usage.prompt_tokens if usage else self.prompt_tokens,
            output_tokens=usage.completion_tokens if usage else self.completion_tokens,
            step=self.name,
            latency_ms=elapsed,
            timestamp=self.start_time,
        )


class Attribute(BaseModel):
    key: str
    value: dict[str, JsonValue]


class Span(BaseModel):
    name: str
    startTimeUnixNano: int
    endTimeUnixNano: int
    attributes: list[Attribute] = []

    def call(self) -> Call | None:
        attrs = {attr.key: next(iter(attr.value.values()), None) for attr in self.attributes}
        if attrs.get("gen_ai.operation.name") not in (
            "chat",
            "text_completion",
            "generate_content",
        ):
            return None
        model = attrs.get("gen_ai.response.model") or attrs.get("gen_ai.request.model")
        prompt = attrs.get("gen_ai.input.messages", attrs.get("gen_ai.prompt"))
        output = attrs.get("gen_ai.output.messages", attrs.get("gen_ai.completion"))
        if not isinstance(model, str) or prompt is None or output is None:
            raise ValueError(f"OpenTelemetry span {self.name!r} lacks model or message content.")
        tokens = [
            attrs.get(key) for key in ("gen_ai.usage.input_tokens", "gen_ai.usage.output_tokens")
        ]
        if any(token is None for token in tokens):
            raise ValueError(f"OpenTelemetry span {self.name!r} lacks token usage.")
        agent = attrs.get("gen_ai.agent.name")
        end = datetime(1970, 1, 1, tzinfo=UTC) + timedelta(
            microseconds=self.endTimeUnixNano // 1000
        )
        return Call(
            prompt=prompt if isinstance(prompt, str) else canonical(prompt),
            output=output,
            model=model,
            input_tokens=int(tokens[0]),
            output_tokens=int(tokens[1]),
            step=f"{agent}/{self.name}" if isinstance(agent, str) else self.name,
            latency_ms=(self.endTimeUnixNano - self.startTimeUnixNano) / 1_000_000,
            timestamp=end,
        )


class ScopeSpans(BaseModel):
    spans: list[Span]


class ResourceSpans(BaseModel):
    scopeSpans: list[ScopeSpans]


class Trace(BaseModel):
    resourceSpans: list[ResourceSpans]


def import_calls(source: Path, format: LogFormat, requests: Path | None) -> tuple[list[Call], int]:
    if format == LogFormat.native:
        return load_rows(source, Call), 0
    if format == LogFormat.openai:
        return [exchange.call() for exchange in load_rows(source, Exchange)], 0
    if format == LogFormat.langsmith:
        runs = load_rows(source, SmithRun)
        calls = [call for run in runs if (call := run.call()) is not None]
        return calls, len(runs) - len(calls)
    if format == LogFormat.otel:
        trace = Trace.model_validate_json(source.read_bytes())
        calls = []
        skipped = 0
        for resource in trace.resourceSpans:
            for scope in resource.scopeSpans:
                for span in scope.spans:
                    call = span.call()
                    if call is None:
                        skipped += 1
                    else:
                        calls.append(call)
        return calls, skipped
    if requests is None:
        raise ValueError("OpenAI Batch import requires --requests with the original batch input.")
    submitted = load_rows(requests, BatchRequest)
    by_id = {row.custom_id: row.body for row in submitted}
    if len(by_id) != len(submitted):
        raise ValueError("OpenAI Batch request IDs must be unique.")
    calls = []
    skipped = 0
    for row in load_rows(source, BatchResult):
        if row.response is None or not 200 <= row.response.status_code < 300:
            skipped += 1
            continue
        if row.custom_id not in by_id:
            raise ValueError("A successful batch response has no matching request ID.")
        calls.append(
            Exchange(
                request=by_id.pop(row.custom_id),
                response=Response.model_validate(row.response.body),
            ).call()
        )
    return calls, skipped
