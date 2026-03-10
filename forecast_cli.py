from __future__ import annotations

import os
import subprocess
import sys
import warnings
from pathlib import Path


def _local_python_candidates(project_root: Path) -> list[Path]:
    return [
        project_root / ".venv" / "bin" / "python",
        project_root / ".venv" / "Scripts" / "python.exe",
    ]


def _find_local_python(project_root: Path) -> Path | None:
    for candidate in _local_python_candidates(project_root):
        if candidate.exists():
            return candidate
    return None


def ensure_local_venv(argv: list[str] | None = None) -> None:
    project_root = Path(__file__).resolve().parent
    local_python = _find_local_python(project_root)
    if local_python is None:
        return

    current_prefix = Path(sys.prefix).resolve()
    target_prefix = local_python.parent.parent.resolve()
    if current_prefix == target_prefix:
        return

    args = argv if argv is not None else sys.argv[1:]
    os.execv(str(local_python), [str(local_python), "-m", "forecast_cli", *args])


def _detect_physical_cpu_count() -> str:
    configured = os.environ.get("LOKY_MAX_CPU_COUNT")
    if configured:
        return configured

    if sys.platform == "darwin":
        try:
            detected = subprocess.check_output(
                ["sysctl", "-n", "hw.physicalcpu"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            if detected.isdigit() and int(detected) > 0:
                return detected
        except Exception:
            logical_cores = os.cpu_count() or 1
            return str(logical_cores - 1 if logical_cores > 1 else 1)

    return str(os.cpu_count() or 1)


def run() -> int:
    project_root = Path(__file__).resolve().parent
    ensure_local_venv()
    mplconfig_dir = project_root / "artifacts" / "mplconfig"
    mplconfig_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mplconfig_dir))
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", _detect_physical_cpu_count())
    warnings.filterwarnings(
        "ignore",
        message="Could not find the number of physical cores for the following reason:",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module=r"joblib\.externals\.loky\.backend\.context",
    )
    from cepea_forecast.cli import main

    return main()


if __name__ == "__main__":
    raise SystemExit(run())
