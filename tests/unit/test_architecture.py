"""Enforce the layering rules from AGENTS.md / docs/architecture.md by scanning imports (AST, no deps)."""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "boxing_ai"

# Third-party modules that pure-core packages must never import.
HEAVY_OR_IO = {
    "psycopg",
    "psycopg_pool",
    "fastapi",
    "uvicorn",
    "streamlit",
    "cv2",
    "torch",
    "ultralytics",
    "sqlalchemy",
}

# Layer -> other boxing_ai packages it must NOT import.
# (ontology is a leaf; events sits on it; sequences on events; analytics on sequences.)
FORBIDDEN_INTERNAL = {
    "ontology": {
        "events",
        "sequences",
        "analytics",
        "evaluation",
        "annotations",
        "database",
        "vision",
    },
    "events": {"sequences", "analytics", "evaluation", "annotations", "database", "vision"},
    "sequences": {"analytics", "evaluation", "annotations", "database", "vision"},
    "analytics": {"evaluation", "annotations", "database", "vision"},
    "evaluation": {"annotations", "database", "vision"},
    "annotations": {"sequences", "analytics", "evaluation", "database", "vision"},
    "database": {"analytics", "sequences", "evaluation", "vision"},
}
PURE_CORE = {"ontology", "events", "sequences", "analytics", "evaluation", "annotations"}


def _layer_files(layer: str) -> list[Path]:
    module_file = SRC / f"{layer}.py"
    if module_file.exists():
        return [module_file]
    package_dir = SRC / layer
    return sorted(package_dir.rglob("*.py")) if package_dir.is_dir() else []


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.append(node.module)
    return found


def test_source_tree_is_present():
    assert SRC.is_dir()
    assert _layer_files("ontology") and _layer_files("events")


def test_pure_core_never_imports_io_or_heavy_libraries():
    violations = []
    for layer in PURE_CORE:
        for path in _layer_files(layer):
            for mod in _imports(path):
                if mod.split(".")[0] in HEAVY_OR_IO:
                    violations.append(f"{path.relative_to(SRC)} imports {mod}")
    assert not violations, "\n".join(violations)


def test_layers_only_depend_downwards():
    violations = []
    for layer, forbidden in FORBIDDEN_INTERNAL.items():
        for path in _layer_files(layer):
            for mod in _imports(path):
                parts = mod.split(".")
                if parts[0] == "boxing_ai" and len(parts) > 1 and parts[1] in forbidden:
                    violations.append(f"{layer}: {path.relative_to(SRC)} imports {mod}")
    assert not violations, "\n".join(violations)
