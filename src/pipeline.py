
"""Run the SwiftDrop XGBoost workflow end to end."""

from pathlib import Path
import subprocess
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "raw" / "zomato_cleaned.csv"


def run_step(description, command):
    print(f"\n{'=' * 65}")
    print(f"STEP: {description}")
    print(f"{'=' * 65}")

    subprocess.run(
        command,
        cwd=ROOT_DIR,
        check=True,
    )

    print(f"COMPLETED: {description}")


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    python = sys.executable

    # 1. Load data, preprocess, train and log metrics with MLflow.
    run_step(
        "Train XGBoost pipeline",
        [
            python,
            "-m",
            "src.train_xgboost",
            "--data",
            str(DATA_PATH),
            "--output-dir",
            str(ROOT_DIR / "models"),
        ],
    )

    # 2. Export the trained XGBoost estimator to ONNX.
    run_step(
        "Export XGBoost to ONNX",
        [python, "-m", "src.export_xgboost_onnx"],
    )

    # 3. Compare sklearn and ONNX predictions on the test split.
    run_step(
        "Evaluate ONNX prediction parity",
        [python, "-m", "src.evaluate_xgboost_onnx"],
    )

    # 4. Benchmark ONNX CPU inference latency.
    run_step(
        "Benchmark ONNX inference",
        [python, "-m", "src.benchmark_xgboost_onnx"],
    )

    print("\nSwiftDrop XGBoost workflow completed successfully.")


if __name__ == "__main__":
    main()
