"""Regenerate docs/API_REFERENCE.md from the source code.

    python scripts/build_api_reference.py          write the file
    python scripts/build_api_reference.py --check  exit 1 if the file is out of date

The reference lists every module, constant, class and function with its
signature and docstring, read straight from the code with `ast`, so it cannot
describe something that no longer exists. A test runs the --check mode, which
means a function added without updating the reference fails the test suite.
"""
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "API_REFERENCE.md"
SOURCES = [ROOT / "run.py"] + sorted((ROOT / "src" / "smokeshift").glob("*.py")) + [
    ROOT / "scripts" / "build_api_reference.py", ROOT / "tests" / "test_smokeshift.py"]


def _signature(node, source):
    """`name(arguments)` as written in the source."""
    return f"{node.name}({_arguments(node, source)})"


def _doc(node, indent=""):
    """The docstring as indented markdown text, or a marker if there is none."""
    doc = ast.get_docstring(node)
    if not doc:
        return f"{indent}*(no docstring)*\n"
    return "\n".join(f"{indent}{line}".rstrip() for line in doc.split("\n")) + "\n"


def _constants(tree, source):
    """Top-level names assigned in a module, with their source text when it is short.

    The value is copied from the source file, not re-rendered from the syntax
    tree, so the output is the same on every Python version.
    """
    rows = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and all(isinstance(t, ast.Name) for t in node.targets):
            names = ", ".join(t.id for t in node.targets)
            if names.isupper() or names in {"print"}:
                value = " ".join((ast.get_source_segment(source, node.value) or "").split())
                rows.append((names, value if len(value) <= 90 else value[:87] + "..."))
    return rows


def _arguments(node, source):
    """The argument list of a function exactly as written, on one line."""
    lines = source.splitlines()
    text = " ".join(lines[node.lineno - 1:node.body[0].lineno - 1]) if node.body[0].lineno > node.lineno \
        else lines[node.lineno - 1]
    start = text.index("(", text.index(f"def {node.name}")) + 1
    depth, end = 1, start
    while depth:                                   # find the bracket that closes the argument list
        depth += {"(": 1, "[": 1, "{": 1, ")": -1, "]": -1, "}": -1}.get(text[end], 0)
        end += 1
    return " ".join(text[start:end - 1].split())


def _functions(body, source, prefix=""):
    """Markdown for every function and class in a body, in source order, recursing into classes."""
    parts = []
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            parts.append(f"#### `{prefix}{_signature(node, source)}`\n\n{_doc(node)}")
            nested = [n for n in node.body if isinstance(n, ast.FunctionDef)]
            for inner in nested:
                first_line = (ast.get_docstring(inner) or "").splitlines()[0]
                parts.append(f"- nested helper `{_signature(inner, source)}`: {first_line}\n")
        elif isinstance(node, ast.ClassDef):
            parts.append(f"#### class `{prefix}{node.name}`\n\n{_doc(node)}")
            parts.append(_functions(node.body, source, prefix=f"{node.name}."))
    return "\n".join(parts)


def build():
    """Return the full text of the reference."""
    out = ["# API reference", "",
           "Generated from the source by `python scripts/build_api_reference.py`. Do not edit by hand.",
           "For the narrative explanation of how the pieces fit together, read `CODE_GUIDE.md` first.", ""]
    for path in SOURCES:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        rel = path.relative_to(ROOT).as_posix()
        out += [f"## `{rel}`", "", f"{len(source.splitlines())} lines.", "", _doc(tree)]
        constants = _constants(tree, source)
        if constants:
            out += ["**Constants**", "", "| Name | Value |", "|---|---|"]
            out += [f"| `{name}` | `{value.replace('|', chr(92) + '|')}` |" for name, value in constants]
            out.append("")
        out.append(_functions(tree.body, source))
    return "\n".join(out).rstrip() + "\n"


def main():
    """Write the reference, or with --check compare it with what is on disk."""
    text = build()
    if "--check" in sys.argv:
        current = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if current != text:
            sys.exit("docs/API_REFERENCE.md is out of date: run python scripts/build_api_reference.py")
        print("docs/API_REFERENCE.md is up to date")
        return
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {OUT.relative_to(ROOT)} ({len(text.splitlines())} lines)")


if __name__ == "__main__":
    main()
