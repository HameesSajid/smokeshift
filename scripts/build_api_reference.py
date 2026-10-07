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


def _signature(node):
    """`name(arguments)` as written in the source."""
    return f"{node.name}({ast.unparse(node.args)})"


def _doc(node, indent=""):
    """The docstring as indented markdown text, or a marker if there is none."""
    doc = ast.get_docstring(node)
    if not doc:
        return f"{indent}*(no docstring)*\n"
    return "\n".join(f"{indent}{line}".rstrip() for line in doc.split("\n")) + "\n"


def _constants(tree):
    """Top-level names assigned in a module, with their source text when it is short."""
    rows = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and all(isinstance(t, ast.Name) for t in node.targets):
            names = ", ".join(t.id for t in node.targets)
            if names.isupper() or names in {"print"}:
                value = ast.unparse(node.value)
                rows.append((names, value if len(value) <= 90 else value[:87] + "..."))
    return rows


def _functions(body, prefix=""):
    """Markdown for every function and class in a body, in source order, recursing into classes."""
    parts = []
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            parts.append(f"#### `{prefix}{_signature(node)}`\n\n{_doc(node)}")
            nested = [n for n in node.body if isinstance(n, ast.FunctionDef)]
            for inner in nested:
                parts.append(f"- nested helper `{_signature(inner)}`: {(ast.get_docstring(inner) or '').splitlines()[0]}\n")
        elif isinstance(node, ast.ClassDef):
            parts.append(f"#### class `{prefix}{node.name}`\n\n{_doc(node)}")
            parts.append(_functions(node.body, prefix=f"{node.name}."))
    return "\n".join(parts)


def build():
    """Return the full text of the reference."""
    out = ["# API reference", "",
           "Generated from the source by `python scripts/build_api_reference.py`. Do not edit by hand.",
           "For the narrative explanation of how the pieces fit together, read `CODE_GUIDE.md` first.", ""]
    for path in SOURCES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        rel = path.relative_to(ROOT).as_posix()
        n_lines = len(path.read_text(encoding="utf-8").splitlines())
        out += [f"## `{rel}`", "", f"{n_lines} lines.", "", _doc(tree)]
        constants = _constants(tree)
        if constants:
            out += ["**Constants**", "", "| Name | Value |", "|---|---|"]
            out += [f"| `{name}` | `{value.replace('|', chr(92) + '|')}` |" for name, value in constants]
            out.append("")
        out.append(_functions(tree.body))
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
