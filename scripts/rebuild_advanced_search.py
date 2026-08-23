import ast
import json
from pathlib import Path


NOTEBOOK_PATH = Path(
    "notebooks/11_advanced_search_engine.ipynb"
)

OUTPUT_PATH = Path(
    "src/engine/advanced_search.py"
)


if not NOTEBOOK_PATH.exists():
    raise FileNotFoundError(
        f"Notebook not found: {NOTEBOOK_PATH}"
    )


with NOTEBOOK_PATH.open(
    "r",
    encoding="utf-8",
) as file:
    notebook = json.load(file)


latest_definitions = {}


for cell_number, cell in enumerate(
    notebook.get("cells", []),
    start=1,
):
    if cell.get("cell_type") != "code":
        continue

    source = "".join(
        cell.get("source", [])
    )

    try:
        tree = ast.parse(source)
    except SyntaxError:
        continue

    for node in tree.body:
        if isinstance(
            node,
            (
                ast.ClassDef,
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            ),
        ):
            latest_definitions[node.name] = {
                "cell": cell_number,
                "code": ast.unparse(node),
            }


required_order = [
    "BoundType",
    "TranspositionEntry",
    "ZobristHasher",
    "SearchStats",
    "SearchResult",
    "AdvancedSearchEngine",
    "search_for_time",
]


missing = [
    name
    for name in required_order
    if name not in latest_definitions
]

if missing:
    raise RuntimeError(
        "Required definitions missing: "
        + ", ".join(missing)
    )


module_lines = [
    '"""Production advanced search engine from Notebook 11."""',
    "",
    "from __future__ import annotations",
    "",
    "import math",
    "import random",
    "import time",
    "",
    "from collections import defaultdict",
    "from dataclasses import dataclass, field",
    "from enum import Enum, auto",
    "from typing import (",
    "    Any,",
    "    Callable,",
    "    Dict,",
    "    Hashable,",
    "    List,",
    "    Optional,",
    "    Sequence,",
    "    Tuple,",
    ")",
    "",
    "from .timer import SearchTimeLimit",
    "",
    "",
]


for name in required_order:
    information = latest_definitions[name]

    module_lines.append(
        information["code"]
    )
    module_lines.append("")
    module_lines.append("")


module_lines.extend(
    [
        "# Attach the timed-search function to the engine class.",
        "AdvancedSearchEngine.search_for_time = search_for_time",
        "",
    ]
)


OUTPUT_PATH.write_text(
    "\n".join(module_lines),
    encoding="utf-8",
)


print("Advanced-search module rebuilt.")
print(f"Output: {OUTPUT_PATH}")
print(
    "File size:",
    OUTPUT_PATH.stat().st_size,
    "bytes",
)

print()
print("Definition order:")

for index, name in enumerate(
    required_order,
    start=1,
):
    print(
        f"  {index}. {name} "
        f"(Notebook cell "
        f"{latest_definitions[name]['cell']})"
    )

