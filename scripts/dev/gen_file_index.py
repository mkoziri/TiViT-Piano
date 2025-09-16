"""Purpose:
    Generate ``docs/FILE_INDEX.md`` summarizing module purposes, top-level
    definitions, and CLI arguments across the repository.

Key Functions/Classes:
    - summarize_file(): Builds the Markdown section describing a single source
      file, including key functions/classes and CLI tables.
    - parse_cli(): Extracts ``argparse`` definitions from an abstract syntax
      tree to document command-line interfaces automatically.

CLI:
    Run ``python scripts/dev/gen_file_index.py`` to refresh the Markdown index.
    The script takes no arguments and writes output in-place.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any


def _find_root(start: Path) -> Path:
    """Walk up from *start* until a repository marker is found."""
    for parent in [start] + list(start.parents):
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    raise FileNotFoundError("Could not locate project root")


ROOT = _find_root(Path(__file__).resolve())
DOC_PATH = ROOT / "docs" / "FILE_INDEX.md"
SCRIPT_REL = Path(__file__).resolve().relative_to(ROOT)


def parse_cli(tree: ast.AST) -> list[dict[str, Any]]:
    """Extract argparse ``add_argument`` calls from a module."""
    args: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
        ):
            entry: dict[str, Any] = {"flags": []}
            for a in node.args:
                if isinstance(a, ast.Constant) and isinstance(a.value, str):
                    entry["flags"].append(a.value)
            for kw in node.keywords:
                if kw.arg == "help" and isinstance(kw.value, ast.Constant):
                    entry["help"] = kw.value.value
                elif kw.arg == "default" and isinstance(kw.value, ast.Constant):
                    entry["default"] = kw.value.value
                elif kw.arg == "type":
                    if isinstance(kw.value, ast.Name):
                        entry["type"] = kw.value.id
                    elif isinstance(kw.value, ast.Attribute):
                        entry["type"] = kw.value.attr
                elif kw.arg == "action" and isinstance(kw.value, ast.Constant):
                    entry["type"] = kw.value.value
                elif kw.arg == "dest" and isinstance(kw.value, ast.Constant):
                    entry["dest"] = kw.value.value
                elif kw.arg == "choices" and isinstance(
                    kw.value, (ast.List, ast.Tuple)
                ):
                    choices = []
                    for elt in kw.value.elts:
                        if isinstance(elt, ast.Constant):
                            choices.append(str(elt.value))
                    if choices:
                        entry["choices"] = choices
            if "dest" not in entry and entry["flags"]:
                flag = entry["flags"][0].lstrip("-").replace("-", "_")
                entry["dest"] = flag
            if "type" not in entry:
                entry["type"] = "str"
            args.append(entry)
    return args


def is_model(node: ast.ClassDef) -> bool:
    for base in node.bases:
        if isinstance(base, ast.Attribute) and base.attr == "Module":
            if isinstance(base.value, ast.Name) and base.value.id == "nn":
                return True
            if (
                isinstance(base.value, ast.Attribute)
                and base.value.attr == "nn"
                and isinstance(base.value.value, ast.Name)
                and base.value.value.id == "torch"
            ):
                return True
        if isinstance(base, ast.Name) and base.id == "Module":
            return True
    return False


def list_defs(tree: ast.AST) -> list[str]:
    """List top-level functions and classes with docstring summaries."""
    items: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            doc = ast.get_docstring(node) or ""
            summary = ""
            if is_model(node):
                summary = " — subclass of `torch.nn.Module`"
            if doc:
                summary += ("; " if summary else " — ") + doc.splitlines()[0]
            items.append(f"- class `{node.name}`{summary}")
        elif isinstance(node, ast.FunctionDef):
            doc = ast.get_docstring(node)
            if doc:
                summary = doc.splitlines()[0]
                items.append(f"- `{node.name}()` — {summary}")
    return items


def summarize_file(path: Path) -> str:
    text = path.read_text()
    tree = ast.parse(text)
    doc = ast.get_docstring(tree) or ""
    summary_lines = doc.splitlines()
    purpose = " ".join(summary_lines[:3]) if summary_lines else "N/A"
    rel = path.relative_to(ROOT)
    lines = [
        f"### {rel.as_posix()}",
        f"**Purpose:** {purpose}",
        "",
    ]
    defs = list_defs(tree)
    if defs:
        lines.append("**Key Functions/Classes**")
        lines.extend(defs)
        lines.append("")
    cli = parse_cli(tree)
    if cli:
        lines.append("**CLI**")
        lines.append("| Flag(s) | Dest | Type | Default | Help |")
        lines.append("|---|---|---|---|---|")
        for c in cli:
            flags = ", ".join(c.get("flags", []))
            dest = c.get("dest", "")
            typ = c.get("type", "")
            default = c.get("default", "")
            default_str = str(default)
            help_text = c.get("help", "")
            if choices := c.get("choices"):
                extra = f"Choices: {', '.join(choices)}."
                help_text = f"{help_text} {extra}".strip()
            lines.append(
                f"| `{flags}` | {dest} | {typ} | `{default_str}` | {help_text} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    files: list[Path] = []
    files.extend(sorted((ROOT / "src").rglob("*.py")))
    files.extend(sorted((ROOT / "scripts").rglob("*.py")))
    files.extend(sorted(ROOT.glob("*.py")))
    yaml_files = sorted((ROOT / "configs").rglob("*.yaml"))
    lines = [
        "# File Index",
        "",
        f"Generated by `{SCRIPT_REL.as_posix()}`.",
        "",
    ]
    for path in files:
        rel = path.relative_to(ROOT)
        lines.append(summarize_file(path))
    for yml in yaml_files:
        rel = yml.relative_to(ROOT).as_posix()
        text = yml.read_text().splitlines()
        keys = [
            line.split(":")[0]
            for line in text
            if ":" in line and not line.startswith(" ")
        ]
        lines.append(f"### {rel}\nTop-level keys: {', '.join(keys)}\n")
    DOC_PATH.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
