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


# Store the last occurrence of every top-level definition.
latest_definitions = {}

# Preserve required imports without repeating them.
imports = {}

# Preserve uppercase constants such as INF or MATE_SCORE.
constants = {}


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
        # Ignore cells containing notebook-only syntax.
        continue

    for node in tree.body:

        if isinstance(
            node,
            (
                ast.Import,
                ast.ImportFrom,
            ),
        ):
            code = ast.unparse(node)
            imports[code] = code

        elif isinstance(
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
                "kind": type(node).__name__,
            }

        elif isinstance(
            node,
            (
                ast.Assign,
                ast.AnnAssign,
            ),
        ):
            targets = []

            if isinstance(node, ast.Assign):
                targets = node.targets
            else:
                targets = [node.target]

            for target in targets:
                if (
                    isinstance(target, ast.Name)
                    and target.id.isupper()
                ):
                    constants[target.id] = ast.unparse(node)


required_names = [
    "ZobristHasher",
    "SearchStats",
    "SearchResult",
    "TranspositionEntry",
    "TTEntry",
    "TranspositionFlag",
    "BoundType",
    "AdvancedSearchEngine",
    "search_for_time",
]


# Include all definitions because helper records may use names
# not listed above. The last version of each name is retained.
ordered_definitions = sorted(
    latest_definitions.items(),
    key=lambda item: item[1]["cell"],
)


module_lines = [
    '"""Advanced game-tree search engine generated from Notebook 11."""',
    "",
    "from __future__ import annotations",
    "",
]

# Add imports gathered from the notebook.
for import_code in sorted(imports):
    if import_code == "from __future__ import annotations":
        continue

    module_lines.append(import_code)

# The timer is maintained in its own module.
module_lines.extend(
    [
        "",
        "from .timer import SearchTimeLimit",
        "",
    ]
)

# Add module-level constants.
for constant_name in sorted(constants):
    module_lines.append(
        constants[constant_name]
    )

if constants:
    module_lines.append("")

# Add the last version of every function and class.
for name, information in ordered_definitions:
    # SearchTimeLimit now belongs in timer.py.
    if name == "SearchTimeLimit":
        continue

    # Benchmark utilities now belong in benchmark.py.
    if name in {
        "EngineBenchmarkResult",
        "benchmark_engine",
    }:
        continue

    module_lines.append(
        information["code"]
    )
    module_lines.append("")
    module_lines.append("")


# Notebook 11 attached timed search dynamically.
# Preserve that behavior when search_for_time is top-level.
if (
    "search_for_time" in latest_definitions
    and "AdvancedSearchEngine" in latest_definitions
):
    module_lines.extend(
        [
            "# Attach the time-limited search method.",
            "AdvancedSearchEngine.search_for_time = search_for_time",
            "",
        ]
    )


OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_PATH.write_text(
    "\n".join(module_lines),
    encoding="utf-8",
)


print("Advanced-search module created.")
print(f"Output: {OUTPUT_PATH}")
print(
    "File size:",
    OUTPUT_PATH.stat().st_size,
    "bytes",
)

print()
print("Important definitions found:")

for name in required_names:
    if name in latest_definitions:
        information = latest_definitions[name]

        print(
            f"  FOUND   {name:<24} "
            f"cell {information['cell']}"
        )
    else:
        print(
            f"  missing {name}"
        )
