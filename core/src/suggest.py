from groq import Groq
from pydantic import ValidationError

from .models import Candidate, Suggestion, TaskSpec
from .worker import (
    INVALID_CODE,
    METHODS,
    MODULES,
    NODES,
    SAFE_EXCEPTIONS,
    SAFE_FUNCTIONS,
    validate_code,
)

MODELS = {"simple": "openai/gpt-oss-20b", "hard": "openai/gpt-oss-120b"}
RULES = {
    "ticket-id": (r"\bCT-[0-9]{4}\b", "A ticket ID is CT- followed by exactly four ASCII digits."),
    "iso-date": (
        r"\b[0-9]{4}-[0-9]{2}-[0-9]{2}\b",
        "Matches YYYY-MM-DD text without calendar validation.",
    ),
    "email": (
        r"(?<![\w.+-])[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+",
        "Matches common ASCII email addresses.",
    ),
}
INSTRUCTIONS = f"""
Return a JSON object containing status, code, and assumptions.
For a candidate, define a deterministic Python function named replace with one positional argument,
prompt, the exact string the original LLM would receive. Return JSON-compatible data
matching the task output type. A JSON-encoded string and a JSON object are different.
Module scope accepts imports of re and json without aliases and plain function definitions.
Helpers may be defined at module scope. Functions use positional parameters without defaults.
No decorators, nested functions, private names, attribute writes, or argument unpacking.
Use try/except with named data exceptions. No bare except or finally blocks.
Allowed functions: {", ".join(sorted(SAFE_FUNCTIONS))}.
Allowed data exceptions: {", ".join(sorted(SAFE_EXCEPTIONS))}.
Allowed data methods: {", ".join(sorted(METHODS))}.
Allowed module attributes: {
    "; ".join(f"{module}: {', '.join(sorted(names))}" for module, names in MODULES.items())
}.
Allowed AST nodes: {", ".join(sorted(node.__name__ for node in NODES))}.
Set status to candidate and provide the module without Markdown fences.
For tasks that depend on judgment or accept many valid answers, set status to keep_ai
and leave code empty. Explain which part of the task requires judgment and name a
narrower part that could be verified. Consider the whole task, especially whether it asks
to extract existing text or generate a new answer. Never infer a refusal from a word alone.
If the description lacks enough detail to decide, use unsupported and identify what is missing.
""".strip()


def offline(spec: TaskSpec) -> Candidate:
    rule = spec.rule
    description = spec.description.casefold()
    status = "candidate"
    code = ""
    if rule is None:
        if "extract" in description and ("ticket id" in description and "ct-" in description):
            rule = "ticket-id"
        elif "extract" in description and (
            "yyyy-mm-dd" in description or "iso date" in description
        ):
            rule = "iso-date"
        elif "extract" in description and "first email address" in description:
            rule = "email"
        elif "lowercase" in description and "entire" in description:
            rule = "lowercase"
        elif "trim" in description and ("whitespace" in description or "spaces" in description):
            rule = "trim"

    if rule == "keep-ai":
        status = "keep_ai"
        assumptions = ["The task owner marked this step as requiring a model.", spec.description]
    elif rule in RULES:
        pattern, note = RULES[rule]
        code = (
            "import re\n\ndef replace(prompt):\n"
            f"    match = re.search({pattern!r}, prompt)\n"
            "    return match.group() if match else None\n"
        )
        assumptions = [note, "Returns the first match as a string or null when none is found."]
    elif rule in ("trim", "lowercase"):
        method = {"trim": "strip", "lowercase": "lower"}[rule]
        code = f"def replace(prompt):\n    return prompt.{method}()\n"
        assumptions = [f"Applies str.{method} to the entire prompt."]
    else:
        status = "unsupported"
        assumptions = ["Describe a supported task rule explicitly, or use a Groq model."]
    return Candidate(
        task=spec.task,
        model="rules",
        suggestion=Suggestion(status=status, code=code, assumptions=assumptions),
    )


def generate(spec: TaskSpec, api_key: str, model: str | None = None) -> Candidate:
    model = model or MODELS[spec.complexity]
    options = (
        {"reasoning_effort": "low" if spec.complexity == "simple" else "medium"}
        if model.startswith("openai/gpt-oss-")
        else {}
    )
    messages = [
        {"role": "system", "content": INSTRUCTIONS},
        {"role": "user", "content": spec.description},
    ]
    with Groq(api_key=api_key, max_retries=0, timeout=90) as client:
        for attempt in range(2):
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0,
                max_completion_tokens=8192,
                **options,
            )
            content = response.choices[0].message.content or "" if response.choices else ""
            try:
                suggestion = Suggestion.model_validate_json(content)
                if suggestion.status == "candidate":
                    validate_code(suggestion.code)
                else:
                    suggestion.code = ""
            except (*INVALID_CODE, ValidationError) as error:
                if isinstance(error, ValidationError):
                    reason = "Response did not match the required JSON object."
                elif isinstance(error, RecursionError):
                    reason = "Python source was too deeply nested."
                else:
                    reason = str(error)
                if attempt == 0:
                    messages = [
                        *messages,
                        {"role": "assistant", "content": content},
                        {
                            "role": "user",
                            "content": f"Repair the response for the sandbox. {reason}",
                        },
                    ]
                    continue
                suggestion = Suggestion(
                    status="unsupported",
                    code="",
                    assumptions=[
                        f"Groq did not produce a usable candidate after one repair. {reason}"
                    ],
                )
            break
    return Candidate(task=spec.task, model=model, suggestion=suggestion)
