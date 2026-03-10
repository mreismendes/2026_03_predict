from __future__ import annotations

from pathlib import Path

import forecast_cli


def test_find_local_python_prefers_project_venv(tmp_path: Path) -> None:
    project_root = tmp_path
    local_python = project_root / ".venv" / "bin" / "python"
    local_python.parent.mkdir(parents=True)
    local_python.write_text("", encoding="utf-8")

    found = forecast_cli._find_local_python(project_root)

    assert found == local_python
