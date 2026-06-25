from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    script = (
        Path(__file__).resolve().parents[3]
        / "scripts"
        / "prepare"
        / "inspect_relbench_bundle.py"
    )

    if not script.exists():
        raise FileNotFoundError(
            f"Inspection script not found: {script}"
        )

    runpy.run_path(
        str(script),
        run_name="__main__",
    )
