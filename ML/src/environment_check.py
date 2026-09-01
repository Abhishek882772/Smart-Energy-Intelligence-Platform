"""ML Environment Verification Script.

Validates that the ML research workspace structure, Python runtime,
core scientific/ML libraries, and PyTorch device configurations are correctly set up.
"""

from __future__ import annotations

import importlib.metadata
from pathlib import Path
import sys


def check_directories(ml_root: Path) -> None:
    expected_directories = (
        ml_root / "data" / "raw",
        ml_root / "data" / "processed",
        ml_root / "notebooks",
        ml_root / "src",
        ml_root / "experiments",
        ml_root / "results",
        ml_root / "checkpoints",
        ml_root / "configs",
    )
    missing = [path.relative_to(ml_root) for path in expected_directories if not path.is_dir()]
    if missing:
        missing_str = ", ".join(str(p) for p in missing)
        raise SystemExit(f"[ERROR] ML workspace check failed; missing directories: {missing_str}")


def check_package(package_name: str, import_name: str | None = None) -> tuple[bool, str]:
    if import_name is None:
        import_name = package_name
    try:
        mod = __import__(import_name)
        version = getattr(mod, "__version__", None)
        if version is None:
            version = importlib.metadata.version(package_name)
        return True, str(version)
    except ImportError:
        return False, "NOT INSTALLED"


def main() -> None:
    ml_root = Path(__file__).resolve().parents[1]
    print("=" * 60)
    print("ML ENVIRONMENT & WORKSPACE VERIFICATION")
    print("=" * 60)

    # 1. Directory Structure
    check_directories(ml_root)
    print(f"Workspace root:      {ml_root}")
    print("Directory layout:    OK (all 8 research directories present)")

    # 2. Python Runtime
    python_ver = sys.version.split()[0]
    print(f"Python executable:   {sys.executable}")
    print(f"Python version:      {python_ver}")

    # 3. Core ML & Scientific Libraries
    packages = [
        ("pip", "pip"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("matplotlib", "matplotlib"),
        ("scikit-learn", "sklearn"),
        ("torch", "torch"),
        ("pyyaml", "yaml"),
        ("jupyter", "jupyter"),
        ("ipykernel", "ipykernel"),
    ]

    all_installed = True
    print("\nPackage Status:")
    for pkg_name, import_alias in packages:
        installed, ver = check_package(pkg_name, import_alias)
        status = "OK" if installed else "MISSING"
        print(f"  - {pkg_name:<16} : {ver:<18} [{status}]")
        if not installed:
            all_installed = False

    if not all_installed:
        print("\n[ERROR] One or more required packages are missing. Run: pip install -r ML/requirements.txt")
        sys.exit(1)

    # 4. PyTorch & Device Sanity Check
    import torch

    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")

    print("\nCompute Device Status:")
    print(f"  - CUDA Available:    {cuda_available}")
    print(f"  - Selected Device:   {device}")

    # 5. Tensor Operations Check
    x = torch.tensor([1.0, 2.0, 3.0], device=device)
    y = x * 2.0
    expected = [2.0, 4.0, 6.0]
    if y.tolist() != expected:
        raise SystemExit(f"[ERROR] PyTorch tensor test failed: got {y.tolist()}, expected {expected}")

    print(f"  - PyTorch Tensor:    OK (basic operations functional on {device})")
    print("=" * 60)
    print("[SUCCESS] ML environment is fully configured and ready for research.")
    print("=" * 60)


if __name__ == "__main__":
    main()
