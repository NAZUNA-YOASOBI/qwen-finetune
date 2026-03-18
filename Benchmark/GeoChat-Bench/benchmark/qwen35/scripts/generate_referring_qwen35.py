from __future__ import annotations

import sys
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


sys.path.insert(0, str(_project_root() / "src"))
from geochatbench_generate import run_generation


if __name__ == "__main__":
    run_generation(task="referring", model_family="qwen35")
