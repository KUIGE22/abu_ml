#!/usr/bin/env python3
"""
Notebook API Documentation Generator

This script scans Jupyter notebooks (.ipynb) in the repository to extract
public functions and classes, then generates a comprehensive Markdown
documentation file at docs/API.md with signatures, docstrings (when present),
and usage examples discovered within the notebooks.

Public API rules:
- Top-level function/class names that do not start with an underscore

Example extraction rules:
- Search subsequent code cells in the same notebook for the first line that
  appears to call the function ("name(") or instantiate the class
  ("name(") and include that as an example snippet.

Limitations:
- Notebooks are not importable modules by default. Usage examples are based
  on code cells within the notebooks themselves. To use a function/class in
  a reusable way, consider moving it into a .py module or executing the
  defining cell in a notebook session.
"""

from __future__ import annotations

import ast
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


NOTEBOOK_EXTENSIONS = {".ipynb"}


@dataclass
class ExampleSnippet:
    code_lines: List[str] = field(default_factory=list)
    cell_index: int = -1


@dataclass
class MethodDoc:
    name: str
    signature: str
    docstring: Optional[str]


@dataclass
class ApiItem:
    name: str
    kind: str  # "function" or "class"
    notebook_path: Path
    cell_index: int
    signature: str
    docstring: Optional[str]
    examples: List[ExampleSnippet] = field(default_factory=list)
    methods: List[MethodDoc] = field(default_factory=list)  # for class items


def iter_notebooks(root: Path) -> Iterable[Path]:
    for directory_path, _dirnames, filenames in os.walk(root):
        for filename in filenames:
            file_path = Path(directory_path) / filename
            if file_path.suffix in NOTEBOOK_EXTENSIONS:
                yield file_path


def load_notebook_cells(notebook_path: Path) -> List[Dict]:
    with notebook_path.open("r", encoding="utf-8") as f:
        nb = json.load(f)
    cells = nb.get("cells", [])
    return cells


def build_function_signature(node: ast.FunctionDef) -> str:
    args = node.args

    def default_to_str(default_node: Optional[ast.AST]) -> str:
        if default_node is None:
            return ""
        try:
            # ast.unparse is available in py3.9+
            return ast.unparse(default_node)  # type: ignore[attr-defined]
        except Exception:
            return "…"

    parts: List[str] = []

    # Positional-only args (py3.8+)
    for index, arg in enumerate(getattr(args, "posonlyargs", [])):
        parts.append(arg.arg)
    if getattr(args, "posonlyargs", []):
        parts.append("/")

    # Positional-or-keyword args
    num_defaults = len(args.defaults)
    num_args = len(args.args)
    first_default_index = num_args - num_defaults
    for index, arg in enumerate(args.args):
        if index >= first_default_index:
            default_index = index - first_default_index
            default_repr = default_to_str(args.defaults[default_index])
            parts.append(f"{arg.arg}={default_repr}")
        else:
            parts.append(arg.arg)

    # Varargs
    if args.vararg is not None:
        parts.append(f"*{args.vararg.arg}")
    else:
        # If there are kwonlyargs but no vararg, include bare *
        if args.kwonlyargs:
            parts.append("*")

    # Keyword-only args
    for index, kwarg in enumerate(args.kwonlyargs):
        default_repr = default_to_str(args.kw_defaults[index])
        if args.kw_defaults[index] is None:
            parts.append(kwarg.arg)
        else:
            parts.append(f"{kwarg.arg}={default_repr}")

    # Kwargs
    if args.kwarg is not None:
        parts.append(f"**{args.kwarg.arg}")

    # Return annotation
    return_annotation: Optional[str] = None
    if node.returns is not None:
        try:
            return_annotation = ast.unparse(node.returns)  # type: ignore[attr-defined]
        except Exception:
            return_annotation = ""

    joined_params = ", ".join(parts)
    sig = f"{node.name}({joined_params})"
    if return_annotation:
        sig += f" -> {return_annotation}"
    return sig


def build_class_signature(node: ast.ClassDef) -> str:
    bases: List[str] = []
    for base in node.bases:
        try:
            bases.append(ast.unparse(base))  # type: ignore[attr-defined]
        except Exception:
            bases.append("…")
    base_str = f"({', '.join(bases)})" if bases else ""
    return f"{node.name}{base_str}"


def extract_methods_from_class(node: ast.ClassDef) -> List[MethodDoc]:
    methods: List[MethodDoc] = []
    for child in node.body:
        if isinstance(child, ast.FunctionDef) and not child.name.startswith("_"):
            signature = build_function_signature(child)
            docstring = ast.get_docstring(child)
            methods.append(MethodDoc(name=child.name, signature=signature, docstring=docstring))
    return methods


def extract_api_items_from_cells(notebook_path: Path, cells: List[Dict]) -> List[ApiItem]:
    api_items: List[ApiItem] = []
    # First pass: find definitions
    for cell_index, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        source_lines: List[str] = cell.get("source", [])
        try:
            source_code = "".join(source_lines)
            module = ast.parse(source_code)
        except Exception:
            continue
        for node in module.body:
            if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
                signature = build_function_signature(node)
                docstring = ast.get_docstring(node)
                api_items.append(
                    ApiItem(
                        name=node.name,
                        kind="function",
                        notebook_path=notebook_path,
                        cell_index=cell_index,
                        signature=signature,
                        docstring=docstring,
                    )
                )
            elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                signature = build_class_signature(node)
                docstring = ast.get_docstring(node)
                methods = extract_methods_from_class(node)
                api_items.append(
                    ApiItem(
                        name=node.name,
                        kind="class",
                        notebook_path=notebook_path,
                        cell_index=cell_index,
                        signature=signature,
                        docstring=docstring,
                        methods=methods,
                    )
                )

    # Second pass: collect examples from subsequent cells
    for item in api_items:
        pattern = re.compile(rf"\b{re.escape(item.name)}\s*\(")
        for search_cell_index in range(item.cell_index + 1, len(cells)):
            cell = cells[search_cell_index]
            if cell.get("cell_type") != "code":
                continue
            source_lines: List[str] = cell.get("source", [])
            for line in source_lines:
                if pattern.search(line) and not line.lstrip().startswith(("def ", "class ")):
                    # Capture a small snippet around the first match
                    snippet = ExampleSnippet(code_lines=[line.rstrip()], cell_index=search_cell_index)
                    item.examples.append(snippet)
                    break
            if item.examples:
                break

    return api_items


def group_items_by_notebook(items: List[ApiItem]) -> Dict[Path, List[ApiItem]]:
    grouped: Dict[Path, List[ApiItem]] = {}
    for item in items:
        grouped.setdefault(item.notebook_path, []).append(item)
    # Sort for stable output
    for path, group in grouped.items():
        group.sort(key=lambda i: (i.kind, i.name.lower()))
    return grouped


def format_markdown(grouped: Dict[Path, List[ApiItem]]) -> str:
    lines: List[str] = []
    lines.append("# API Documentation")
    lines.append("")
    lines.append(
        "This document summarizes public functions and classes discovered in the Jupyter notebooks. "
        "Public items are top-level names that do not start with an underscore."
    )
    lines.append("")
    lines.append("Note: These APIs live in notebooks. To use them, execute the defining cell in a notebook, or refactor them into a Python module.")
    lines.append("")

    for notebook_path, items in sorted(grouped.items(), key=lambda kv: str(kv[0]).lower()):
        rel_path = os.path.relpath(str(notebook_path), start=str(Path.cwd()))
        lines.append(f"## {rel_path}")
        lines.append("")
        if not items:
            lines.append("No public functions or classes found.")
            lines.append("")
            continue
        for item in items:
            lines.append(f"### {item.name}")
            lines.append("")
            kind_label = "Function" if item.kind == "function" else "Class"
            lines.append(f"- Kind: {kind_label}")
            lines.append(f"- Defined in cell: {item.cell_index}")
            lines.append(f"- Signature: `{item.signature}`")
            if item.docstring:
                lines.append("")
                lines.append("Description:")
                lines.append("")
                for para_line in item.docstring.strip().splitlines():
                    lines.append(f"> {para_line}")
            if item.kind == "class" and item.methods:
                lines.append("")
                lines.append("Methods:")
                lines.append("")
                for method in item.methods:
                    lines.append(f"- `{method.signature}`")
                    if method.docstring:
                        lines.append(f"  - {method.docstring.strip().splitlines()[0]}")
            if item.examples:
                lines.append("")
                lines.append("Example:")
                lines.append("")
                # Only show the first example for brevity
                example = item.examples[0]
                lines.append("```python")
                for ex_line in example.code_lines:
                    lines.append(ex_line)
                lines.append("```")
            lines.append("")
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    repository_root = Path(__file__).resolve().parent.parent
    notebooks: List[Path] = list(iter_notebooks(repository_root))
    all_items: List[ApiItem] = []
    for notebook_path in notebooks:
        try:
            cells = load_notebook_cells(notebook_path)
        except Exception:
            continue
        items = extract_api_items_from_cells(notebook_path, cells)
        all_items.extend(items)

    grouped = group_items_by_notebook(all_items)

    docs_dir = repository_root / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    output_path = docs_dir / "API.md"
    markdown = format_markdown(grouped)
    output_path.write_text(markdown, encoding="utf-8")

    print(f"Generated {output_path} with {len(all_items)} API item(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

