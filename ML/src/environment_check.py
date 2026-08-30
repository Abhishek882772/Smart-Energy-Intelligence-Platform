"""Confirm that the initial ML workspace can run without third-party packages."""

from pathlib import Path
import sys


def main() -> None:
    ml_root = Path(__file__).resolve().parents[1]
    expected_directories = (
        ml_root / "data",
        ml_root / "notebooks",
        ml_root / "src",
        ml_root / "experiments",
        ml_root / "results",
        ml_root / "checkpoints",
        ml_root / "configs",
    )
    missing = [path.relative_to(ml_root) for path in expected_directories if not path.is_dir()]
    if missing:
        raise SystemExit(f"ML workspace check failed; missing directories: {', '.join(map(str, missing))}")

    print(f"ML environment check passed with Python {sys.version.split()[0]}")
    print(f"Workspace: {ml_root}")
    print("No third-party packages are required for this initial check.")


if __name__ == "__main__":
    main()
