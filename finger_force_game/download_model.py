"""Download the official MediaPipe Hand Landmarker model."""

from pathlib import Path
import sys
import urllib.request


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "hand_landmarker.task"
TEMP_PATH = MODEL_DIR / "hand_landmarker.download"

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/"
    "hand_landmarker.task"
)

MINIMUM_EXPECTED_SIZE = 1_000_000


def main() -> int:
    """Download the hand model if it is not already available."""

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if (
        MODEL_PATH.exists()
        and MODEL_PATH.stat().st_size >= MINIMUM_EXPECTED_SIZE
    ):
        print(f"[OK] Hand model already exists: {MODEL_PATH.name}")
        return 0

    MODEL_PATH.unlink(missing_ok=True)
    TEMP_PATH.unlink(missing_ok=True)

    print("Downloading the official MediaPipe hand model...")
    print("Internet is only required during the first setup.")

    request = urllib.request.Request(
        MODEL_URL,
        headers={
            "User-Agent": "Mozilla/5.0 FingerForce/1.0"
        },
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=120
        ) as response:
            total = response.headers.get("Content-Length")
            downloaded = 0

            with TEMP_PATH.open("wb") as output_file:
                while True:
                    chunk = response.read(1024 * 1024)

                    if not chunk:
                        break

                    output_file.write(chunk)
                    downloaded += len(chunk)

                    if total and total.isdigit():
                        percentage = (
                            downloaded * 100 // int(total)
                        )

                        print(
                            f"Downloaded: {percentage}%",
                            end="\r",
                            flush=True,
                        )

        print(" " * 30, end="\r")

        if (
            not TEMP_PATH.exists()
            or TEMP_PATH.stat().st_size
            < MINIMUM_EXPECTED_SIZE
        ):
            raise RuntimeError(
                "The downloaded model file is incomplete."
            )

        TEMP_PATH.replace(MODEL_PATH)

        print(f"[OK] Model saved to: {MODEL_PATH}")
        return 0

    except Exception as error:
        TEMP_PATH.unlink(missing_ok=True)

        print("[ERROR] The hand model could not be downloaded.")
        print(f"Reason: {error}")
        print("Check your internet connection and run again.")

        return 1


if __name__ == "__main__":
    sys.exit(main())