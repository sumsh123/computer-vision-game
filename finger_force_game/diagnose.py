"""Check whether Finger Force is ready to run."""

from pathlib import Path
import platform
import sys


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = (
    BASE_DIR
    / "models"
    / "hand_landmarker.task"
)


def check_import(module_name: str) -> bool:
    """Import one package and display its version."""

    try:
        module = __import__(module_name)

        version = getattr(
            module,
            "__version__",
            "installed",
        )

        print(f"[OK] {module_name}: {version}")
        return True

    except Exception as error:
        print(f"[ERROR] {module_name}: {error}")
        return False


def main() -> int:
    print()
    print("Finger Force setup check")
    print("=" * 40)

    print(f"Python: {sys.version.split()[0]}")
    print(f"Executable: {sys.executable}")
    print(f"System: {platform.platform()}")

    passed = True

    if sys.version_info[:2] != (3, 12):
        print("[ERROR] This project must use Python 3.12.")
        passed = False
    else:
        print("[OK] Correct Python version")

    for package in (
        "numpy",
        "cv2",
        "mediapipe",
        "pygame",
    ):
        if not check_import(package):
            passed = False

    if (
        MODEL_PATH.exists()
        and MODEL_PATH.stat().st_size >= 1_000_000
    ):
        print(f"[OK] Hand model: {MODEL_PATH.name}")
    else:
        print("[ERROR] Hand model is missing or incomplete.")
        passed = False

    print("=" * 40)

    if passed:
        print("SETUP PASSED. The game is ready.")
        return 0

    print("SETUP FAILED. Read the errors above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())