"""Candidate runtime. The child starts with -I -S and a reduced environment."""

import __future__

import ast
import builtins
import json
import re
import sys
from time import perf_counter_ns

MEMORY_BYTES = 128 * 1024 * 1024
MAX_CODE = 64 * 1024
MAX_INPUT = 1024 * 1024
MAX_OUTPUT = 1024 * 1024
SAFE_FUNCTIONS = {
    name: getattr(builtins, name)
    for name in (
        "abs",
        "all",
        "any",
        "bool",
        "dict",
        "enumerate",
        "float",
        "int",
        "isinstance",
        "len",
        "list",
        "max",
        "min",
        "range",
        "reversed",
        "round",
        "set",
        "sorted",
        "str",
        "sum",
        "tuple",
        "zip",
    )
}
SAFE_EXCEPTIONS = {
    cls.__name__: cls for cls in (ValueError, TypeError, KeyError, IndexError, ArithmeticError)
}
MODULES = {
    "re": {
        "compile",
        "search",
        "fullmatch",
        "match",
        "findall",
        "finditer",
        "split",
        "sub",
        "escape",
        "ASCII",
        "IGNORECASE",
        "I",
        "MULTILINE",
        "DOTALL",
    },
    "json": {"loads", "dumps"},
}
METHODS = {
    "append",
    "add",
    "capitalize",
    "casefold",
    "count",
    "copy",
    "end",
    "endswith",
    "extend",
    "find",
    "findall",
    "finditer",
    "fullmatch",
    "get",
    "group",
    "groups",
    "groupdict",
    "index",
    "insert",
    "isalnum",
    "isalpha",
    "isascii",
    "isdecimal",
    "isdigit",
    "islower",
    "isspace",
    "isupper",
    "items",
    "join",
    "keys",
    "lower",
    "lstrip",
    "match",
    "partition",
    "pop",
    "remove",
    "removeprefix",
    "removesuffix",
    "replace",
    "rfind",
    "rpartition",
    "rsplit",
    "rstrip",
    "search",
    "setdefault",
    "split",
    "splitlines",
    "sort",
    "span",
    "start",
    "startswith",
    "strip",
    "sub",
    "title",
    "update",
    "upper",
    "values",
    "zfill",
}
NODES = {
    ast.Module,
    ast.FunctionDef,
    ast.arguments,
    ast.arg,
    ast.Import,
    ast.alias,
    ast.Return,
    ast.Assign,
    ast.AnnAssign,
    ast.AugAssign,
    ast.If,
    ast.IfExp,
    ast.For,
    ast.While,
    ast.Try,
    ast.ExceptHandler,
    ast.Break,
    ast.Continue,
    ast.Pass,
    ast.Expr,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
    ast.comprehension,
    ast.Name,
    ast.Load,
    ast.Store,
    ast.Constant,
    ast.JoinedStr,
    ast.FormattedValue,
    ast.Attribute,
    ast.Call,
    ast.keyword,
    ast.Subscript,
    ast.Slice,
    ast.Tuple,
    ast.List,
    ast.Dict,
    ast.Set,
    ast.Compare,
    ast.BinOp,
    ast.BoolOp,
    ast.UnaryOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.BitOr,
    ast.UAdd,
    ast.USub,
    ast.Not,
    ast.And,
    ast.Or,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.In,
    ast.NotIn,
    ast.Is,
    ast.IsNot,
}


class BlockedCode(Exception):
    pass


INVALID_CODE = (BlockedCode, SyntaxError, RecursionError)


def validate_code(code: str):
    if len(code.encode()) > MAX_CODE:
        raise BlockedCode("Candidate source exceeds 64 KiB.")
    tree = ast.parse(code, filename="<candidate>")
    if any(not isinstance(node, (ast.FunctionDef, ast.Import)) for node in tree.body):
        raise BlockedCode(
            "Module scope only permits function definitions and imports of re or json."
        )
    definitions = {node for node in tree.body if isinstance(node, ast.FunctionDef)}
    names = {function.name for function in definitions}
    if "replace" not in names or len(names) != len(definitions):
        raise BlockedCode("Define replace and give each helper a unique name.")
    reserved = SAFE_FUNCTIONS.keys() | SAFE_EXCEPTIONS.keys() | MODULES.keys()
    callable_names = SAFE_FUNCTIONS.keys() | names
    for function in definitions:
        args = function.args
        if (
            function.name.startswith("_")
            or function.name in reserved
            or args.kwonlyargs
            or args.kwarg
            or args.vararg
            or args.defaults
            or function.decorator_list
            or getattr(function, "type_params", ())
        ):
            raise BlockedCode("Use plain functions with positional parameters and no decorators.")
        if function.name == "replace" and len(args.posonlyargs + args.args) != 1:
            raise BlockedCode("replace must take one positional argument.")
    for node in ast.walk(tree):
        if type(node) not in NODES:
            raise BlockedCode(f"Python syntax {type(node).__name__} is outside the sandbox subset.")
        if isinstance(node, ast.Import) and any(
            alias.name not in MODULES or alias.asname for alias in node.names
        ):
            raise BlockedCode("The sandbox only permits import re and import json without aliases.")
        if isinstance(node, ast.FunctionDef) and node not in definitions:
            raise BlockedCode("Nested functions are unavailable in the sandbox.")
        if isinstance(node, ast.Try) and node.finalbody:
            raise BlockedCode("The sandbox permits try/except without finally.")
        if isinstance(node, ast.ExceptHandler):
            types = node.type.elts if isinstance(node.type, ast.Tuple) else [node.type]
            if not types or any(
                not isinstance(item, ast.Name) or item.id not in SAFE_EXCEPTIONS for item in types
            ):
                raise BlockedCode("Catch a named, allowlisted data exception.")
        if isinstance(node, ast.Name) and node.id.startswith("_"):
            raise BlockedCode("Private Python names are unavailable in the sandbox.")
        if isinstance(node, ast.Attribute):
            allowed = (
                MODULES.get(node.value.id, METHODS) if isinstance(node.value, ast.Name) else METHODS
            )
            if node.attr not in allowed or not isinstance(node.ctx, ast.Load):
                raise BlockedCode(f"Attribute {node.attr!r} is unavailable in the sandbox.")
        if isinstance(node, ast.Call) and (
            not isinstance(node.func, (ast.Name, ast.Attribute))
            or (isinstance(node.func, ast.Name) and node.func.id not in callable_names)
            or any(keyword.arg is None or keyword.arg.startswith("_") for keyword in node.keywords)
        ):
            raise BlockedCode("Only allowlisted functions and data methods can be called.")
    return compile(tree, "<candidate>", "exec", flags=__future__.annotations.compiler_flag)


def safe_import(name: str, *args):
    if name == "re":
        return re
    if name == "json":
        return json
    raise BlockedCode("The module is unavailable in the sandbox.")


def deny_host_access(event: str, args: tuple):
    if event == "open" or event.startswith(("socket.", "subprocess.", "os.", "ctypes.")):
        raise BlockedCode("Host access is unavailable in the sandbox.")


def set_memory_limit():
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class BasicLimits(ctypes.Structure):
            _fields_ = [
                ("per_process_time", ctypes.c_longlong),
                ("per_job_time", ctypes.c_longlong),
                ("flags", wintypes.DWORD),
                ("min_working_set", ctypes.c_size_t),
                ("max_working_set", ctypes.c_size_t),
                ("active_processes", wintypes.DWORD),
                ("affinity", ctypes.c_size_t),
                ("priority", wintypes.DWORD),
                ("scheduling", wintypes.DWORD),
            ]

        class ExtendedLimits(ctypes.Structure):
            _fields_ = [
                ("basic", BasicLimits),
                ("io_counters", ctypes.c_ulonglong * 6),
                ("process_memory", ctypes.c_size_t),
                ("job_memory", ctypes.c_size_t),
                ("peak_process_memory", ctypes.c_size_t),
                ("peak_job_memory", ctypes.c_size_t),
            ]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel.CreateJobObjectW.restype = wintypes.HANDLE
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        kernel.SetInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        job = kernel.CreateJobObjectW(None, None)
        limits = ExtendedLimits()
        limits.basic.flags = 0x100  # JOB_OBJECT_LIMIT_PROCESS_MEMORY
        limits.process_memory = MEMORY_BYTES
        if not (
            job
            and kernel.SetInformationJobObject(job, 9, ctypes.byref(limits), ctypes.sizeof(limits))
            and kernel.AssignProcessToJobObject(job, kernel.GetCurrentProcess())
        ):
            raise ctypes.WinError(ctypes.get_last_error())
    else:
        import resource

        _, hard = resource.getrlimit(resource.RLIMIT_AS)
        limit = MEMORY_BYTES if hard == resource.RLIM_INFINITY else min(MEMORY_BYTES, hard)
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))


def main():
    try:
        if sys.platform != "darwin":
            set_memory_limit()
    except (OSError, ValueError, ImportError) as error:
        print(
            json.dumps(
                {
                    "error": f"Memory isolation unavailable: {type(error).__name__}",
                    "isolation_unavailable": True,
                }
            )
        )
        return

    request = json.load(sys.stdin)
    try:
        compiled = validate_code(request["code"])
        if len(request["input"].encode()) > MAX_INPUT:
            raise BlockedCode("Input exceeds the 1 MiB sandbox limit.")
        namespace = {
            "__builtins__": {**SAFE_FUNCTIONS, **SAFE_EXCEPTIONS, "__import__": safe_import}
        }
        sys.addaudithook(deny_host_access)
        exec(compiled, namespace)
        start = perf_counter_ns()
        value = namespace["replace"](request["input"])
        elapsed = (perf_counter_ns() - start) / 1_000_000
        result = json.dumps({"value": value, "elapsed_ms": elapsed}, allow_nan=False)
        if len(result) > MAX_OUTPUT:
            raise BlockedCode("Output exceeds the 1 MiB sandbox limit.")
    except MemoryError:
        result = json.dumps({"error": "Candidate exceeded the 128 MiB memory limit."})
    except Exception as error:
        result = json.dumps({"error": f"{type(error).__name__}: {str(error)[:250]}"})
    print(result)


if __name__ == "__main__":
    main()
