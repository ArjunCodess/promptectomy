import json
import sys
from time import perf_counter_ns


def main():
    request = json.load(sys.stdin)
    namespace = {"__name__": "_candidate"}
    try:
        exec(compile(request["code"], "<candidate>", "exec"), namespace)
        replace = namespace["replace"]
        start = perf_counter_ns()
        value = replace(request["input"])
        elapsed = (perf_counter_ns() - start) / 1_000_000
        result = json.dumps({"value": value, "elapsed_ms": elapsed}, allow_nan=False)
    except Exception as error:
        result = json.dumps({"error": f"{type(error).__name__}: {str(error)[:500]}"})
    print(result)


if __name__ == "__main__":
    main()
