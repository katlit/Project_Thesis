"""Validate generated notebook JSON and Python cell syntax without executing Colab cells."""

import ast
import json
from pathlib import Path


failures = []
for path in sorted(Path(__file__).parent.glob("*.ipynb")):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        source = "\n".join(
            line for line in source.splitlines()
            if not line.lstrip().startswith(("!", "%"))
        )
        try:
            ast.parse(source)
        except SyntaxError as error:
            failures.append(f"{path.name}, cell {index}: {error}")

if failures:
    raise SystemExit("\n".join(failures))
print("Notebook JSON and Python cell syntax OK")
