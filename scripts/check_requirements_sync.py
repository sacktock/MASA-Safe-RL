"""Check that requirements.txt and pyproject.toml stay in lockstep.

The conda installation flow relies on ``requirements.txt`` while uv/PyPI
consume ``[project.dependencies]`` in ``pyproject.toml``. This script
compares the two with strict string equality so the files cannot drift.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib  # type: ignore[no-redefine]

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
REQUIREMENTS = ROOT / "requirements.txt"

# PEP 503 name normalization: lowercase, collapse runs of -, _ and . to a
# single -.
_NAME_PREFIX = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)(.*)$", re.DOTALL)
_NORMALIZE = re.compile(r"[-_.]+")


def _split_requirement(requirement: str) -> tuple[str, str]:
    """Return (canonical_name, normalized_remainder) for a requirement string.

    The remainder (extras, version specifiers, environment markers) has all
    whitespace stripped so comparison is purely on content, not formatting.
    """
    match = _NAME_PREFIX.match(requirement.strip())
    if not match:
        raise ValueError(f"Could not parse requirement: {requirement!r}")
    name, remainder = match.groups()
    canonical = _NORMALIZE.sub("-", name).lower()
    normalized_remainder = re.sub(r"\s+", "", remainder)
    return canonical, normalized_remainder


def parse_pyproject(path: Path) -> dict[str, str]:
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    dependencies = data.get("project", {}).get("dependencies", [])
    result: dict[str, str] = {}
    for raw in dependencies:
        name, remainder = _split_requirement(raw)
        if name in result:
            raise SystemExit(
                f"Duplicate dependency '{name}' in {path.name} [project.dependencies]"
            )
        result[name] = remainder
    return result


def parse_requirements(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    duplicates: list[str] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.split("#", 1)[0].strip()
        if not stripped:
            continue
        name, remainder = _split_requirement(stripped)
        if name in result:
            duplicates.append(f"  line {lineno}: {name}")
            continue
        result[name] = remainder
    if duplicates:
        raise SystemExit(
            f"Duplicate dependencies in {path.name}:\n" + "\n".join(duplicates)
        )
    return result


def main() -> int:
    pyproject = parse_pyproject(PYPROJECT)
    requirements = parse_requirements(REQUIREMENTS)

    missing = sorted(set(pyproject) - set(requirements))
    extra = sorted(set(requirements) - set(pyproject))
    mismatched = sorted(
        name
        for name in set(pyproject) & set(requirements)
        if pyproject[name] != requirements[name]
    )

    if not (missing or extra or mismatched):
        print(
            f"OK: {REQUIREMENTS.name} and {PYPROJECT.name} "
            f"[project.dependencies] are in sync ({len(pyproject)} dependencies)."
        )
        return 0

    print("requirements.txt and pyproject.toml [project.dependencies] disagree:\n")
    if missing:
        print("Missing from requirements.txt (present in pyproject.toml):")
        for name in missing:
            print(f"  - {name}{pyproject[name]}")
        print()
    if extra:
        print("Extra in requirements.txt (absent from pyproject.toml):")
        for name in extra:
            print(f"  - {name}{requirements[name]}")
        print()
    if mismatched:
        print("Specifier mismatches:")
        for name in mismatched:
            print(
                f"  - {name}: pyproject '{name}{pyproject[name]}' "
                f"!= requirements '{name}{requirements[name]}'"
            )
        print()
    print(
        "Update requirements.txt and [project.dependencies] so they match "
        "exactly (base dependencies only)."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
