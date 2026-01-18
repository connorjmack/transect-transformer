#!/usr/bin/env python3
"""Verify CliffCast project setup.

This script checks that:
1. All required dependencies are installed
2. Configuration files load correctly
3. Package imports work
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_dependencies():
    """Check that required dependencies are installed."""
    print("Checking dependencies...")

    required_modules = [
        "torch",
        "numpy",
        "yaml",
        "omegaconf",
        "pandas",
        "matplotlib",
        "scipy",
        "sklearn",
    ]

    missing = []
    for module in required_modules:
        try:
            __import__(module)
            print(f"  ✓ {module}")
        except ImportError:
            print(f"  ✗ {module} - MISSING")
            missing.append(module)

    if missing:
        print(f"\n⚠ Missing dependencies: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False

    print("✓ All dependencies installed\n")
    return True


def check_package_import():
    """Check that the CliffCast package can be imported."""
    print("Checking package imports...")

    try:
        import src

        print(f"  ✓ src package (version {src.__version__})")

        from src.utils import get_logger, load_config

        print("  ✓ src.utils")

        print("✓ Package imports work\n")
        return True
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False


def check_configs():
    """Check that configuration files load correctly."""
    print("Checking configuration files...")

    from src.utils.config import load_config

    config_files = [
        "configs/default.yaml",
        "configs/phase1_risk_only.yaml",
        "configs/phase2_add_retreat.yaml",
        "configs/phase3_add_collapse.yaml",
        "configs/phase4_full.yaml",
    ]

    all_ok = True
    for config_file in config_files:
        try:
            cfg = load_config(config_file, validate=True)
            print(f"  ✓ {config_file} (d_model={cfg.model.d_model})")
        except Exception as e:
            print(f"  ✗ {config_file} - {e}")
            all_ok = False

    if all_ok:
        print("✓ All configs load successfully\n")
    else:
        print("⚠ Some configs failed to load\n")

    return all_ok


def check_directory_structure():
    """Check that required directories exist."""
    print("Checking directory structure...")

    required_dirs = [
        "src/data",
        "src/models",
        "src/training",
        "src/evaluation",
        "src/visualization",
        "src/inference",
        "src/utils",
        "tests",
        "configs",
        "scripts",
        "notebooks",
    ]

    all_ok = True
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists():
            print(f"  ✓ {dir_path}")
        else:
            print(f"  ✗ {dir_path} - MISSING")
            all_ok = False

    if all_ok:
        print("✓ Directory structure correct\n")
    else:
        print("⚠ Some directories missing\n")

    return all_ok


def main():
    """Run all verification checks."""
    print("=" * 60)
    print("CliffCast Project Setup Verification")
    print("=" * 60)
    print()

    checks = [
        ("Directory Structure", check_directory_structure),
        ("Dependencies", check_dependencies),
        ("Package Import", check_package_import),
        ("Configuration Files", check_configs),
    ]

    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"✗ {name} check failed with error: {e}\n")
            results.append((name, False))

    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)

    all_passed = all(result for _, result in results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8} {name}")

    print()

    if all_passed:
        print("✓ All checks passed! Project setup is complete.")
        print("\nNext steps:")
        print("  1. Prepare data: python scripts/prepare_dataset.py")
        print("  2. Run tests: pytest tests/")
        print("  3. Train model: python train.py --config configs/phase1_risk_only.yaml")
        return 0
    else:
        print("⚠ Some checks failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
