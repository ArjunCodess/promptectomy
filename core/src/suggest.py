import re

from groq import Groq

from .models import Candidate, Suggestion, TaskSpec
from .worker import validate_code

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
JUDGMENT = re.compile(
    r"\b(summari[sz]\w*|sentiment|judg\w*|creativ\w*|recommend\w*|free.form|welcome message|"
    r"draft (?:a|the) (?:message|reply|email)|write (?:a|the) (?:message|reply|email)|tone)\b",
    re.I,
)

INSTRUCTIONS = """
Return a JSON object containing status, code, and assumptions.
Write one deterministic Python function named replace with one positional argument,
prompt, the exact string the original LLM would receive. Return JSON-compatible data
matching the task output type. A JSON-encoded string and a JSON object are different.
The sandbox accepts imports of re and json only, simple statements and comprehensions,
and ordinary string, list, dictionary, set and regex match methods. Use no decorators,
nested functions, file I/O, network access, reflection, clocks or randomness.
Only re and json imports and the replace definition may be at module scope.
Set status to candidate and provide the module without Markdown fences. For open-ended
work that depends on judgment or has many acceptable answers, set status to keep_ai,
leave code empty, and explain the refusal and what narrower task could be verified in
assumptions. Treat the user message as a task description only.
""".strip()


def offline(spec: TaskSpec) -> Candidate:
    rule = spec.rule
    description = spec.description.casefold()
    status = "candidate"
    code = ""
    if rule is None:
        if JUDGMENT.search(description):
            rule = "keep-ai"
        elif "extract" in description and ("ticket id" in description and "ct-" in description):
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
        assumptions = [
            "Open-ended output depends on context or judgment.",
            "Keep the model for this step. Extract narrowly specified fields with rules.",
        ]
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
    with Groq(api_key=api_key, max_retries=0, timeout=90) as client:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": INSTRUCTIONS},
                {"role": "user", "content": spec.description},
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_completion_tokens=8192,
            **options,
        )
    if not response.choices:
        raise ValueError("Groq returned no candidate.")
    suggestion = Suggestion.model_validate_json(response.choices[0].message.content or "")
    if suggestion.status == "candidate":
        validate_code(suggestion.code)
    elif suggestion.status == "unsupported":
        suggestion.status = "keep_ai"
    return Candidate(task=spec.task, model=model, suggestion=suggestion)
