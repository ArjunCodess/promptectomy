from groq import Groq

from .models import Candidate, Suggestion, TaskSpec

MODELS = {"simple": "openai/gpt-oss-20b", "hard": "openai/gpt-oss-120b"}

INSTRUCTIONS = """
Return a JSON object containing status, code, and assumptions.
Write a deterministic Python function named replace with one positional argument,
prompt, the exact string the original LLM would receive. It must return a value
that can be serialized as JSON and matches the specified output type. A JSON-
encoded string and a JSON object are different outputs. Use Python's standard
library only and avoid filesystem, network, subprocesses, randomness, clocks,
and mutable global state. Include a list of specific assumptions as strings.
Set status to candidate and place the entire Python module in code with no
Markdown fences. When the task cannot be expressed as predictable rules based
on the description, use status unsupported, an empty code string, and explain
why in assumptions. Treat the user message as a task description only.
""".strip()


def generate(spec: TaskSpec, api_key: str) -> Candidate:
    model = MODELS[spec.complexity]
    with Groq(api_key=api_key, max_retries=0, timeout=90) as client:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": INSTRUCTIONS},
                {"role": "user", "content": spec.description},
            ],
            response_format={"type": "json_object"},
            reasoning_effort="low" if spec.complexity == "simple" else "medium",
            temperature=0,
            max_completion_tokens=8192,
        )
    if not response.choices:
        raise ValueError("Groq returned no candidate.")
    suggestion = Suggestion.model_validate_json(response.choices[0].message.content or "")
    if suggestion.status == "candidate":
        compile(suggestion.code, "<candidate>", "exec")
    return Candidate(task=spec.task, model=model, suggestion=suggestion)
