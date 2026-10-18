from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.data_loader import load_dataset
from src.preprocessing import TARGET, make_preprocessor, prepare_model_frame


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = ROOT_DIR / "data" / "raw" / "zomato_cleaned.csv"
DEFAULT_PIPELINE_PATH = ROOT_DIR / "models" / "xgboost_pipeline.joblib"
DEFAULT_ONNX_PATH = ROOT_DIR / "models" / "xgboost.onnx"
DEFAULT_REPORT_PATH = ROOT_DIR / "reports" / "xgboost_onnx_evaluation.json"


def calculate_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def measure_latency_ms(predict_fn, X, warmup: int = 20, iterations: int = 100) -> dict:
    """Measure single-row prediction latency; preprocessing is excluded."""
    if len(X) == 0:
        raise ValueError("Cannot benchmark an empty dataset.")

    sample_count = min(100, len(X))
    samples = X[:sample_count]

    for i in range(min(warmup, sample_count)):
        predict_fn(samples[i:i + 1])

    latencies = []
    for i in range(iterations):
        row = samples[i % sample_count:i % sample_count + 1]
        start = time.perf_counter()
        predict_fn(row)
        latencies.append((time.perf_counter() - start) * 1000)

    return {
        "measurement": "single-row model inference; preprocessing excluded",
        "iterations": iterations,
        "mean_ms": float(np.mean(latencies)),
        "median_ms": float(np.median(latencies)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
    }


def evaluate(
    data_path: Path = DEFAULT_DATA_PATH,
    pipeline_path: Path = DEFAULT_PIPELINE_PATH,
    onnx_path: Path = DEFAULT_ONNX_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
) -> dict:
    for path in (pipeline_path, onnx_path):
        if not path.exists():
            raise FileNotFoundError(f"Required model file not found: {path}")

    raw_df = load_dataset(str(data_path) if data_path.exists() else None)
    df = prepare_model_frame(raw_df)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # Reproduce the same deterministic 60/20/20 split used during training.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    pipeline = joblib.load(pipeline_path)
    preprocessor = pipeline.named_steps["preprocessor"]

    X_test_transformed = preprocessor.transform(X_test)
    if hasattr(X_test_transformed, "toarray"):
        X_test_transformed = X_test_transformed.toarray()
    X_test_transformed = np.asarray(X_test_transformed, dtype=np.float32)

    y_test_np = np.asarray(y_test, dtype=np.float64)
    sklearn_predictions = np.asarray(pipeline.predict(X_test), dtype=np.float64)

    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    def onnx_predict(batch):
        output = session.run(
            [output_name],
            {input_name: np.asarray(batch, dtype=np.float32)},
        )[0]
        return np.asarray(output).reshape(-1)

    onnx_predictions = onnx_predict(X_test_transformed).astype(np.float64)

    sklearn_metrics = calculate_metrics(y_test_np, sklearn_predictions)
    onnx_metrics = calculate_metrics(y_test_np, onnx_predictions)

    max_abs_diff = float(np.max(np.abs(sklearn_predictions - onnx_predictions)))
    mean_abs_diff = float(np.mean(np.abs(sklearn_predictions - onnx_predictions)))
    parity_pass = bool(
        np.allclose(sklearn_predictions, onnx_predictions, rtol=1e-4, atol=1e-3)
    )

    X_benchmark = X_test_transformed
    sklearn_latency = measure_latency_ms(
        lambda batch: pipeline.named_steps["model"].predict(batch),
        X_benchmark,
    )
    onnx_latency = measure_latency_ms(onnx_predict, X_benchmark)

    report = {
        "dataset": str(data_path),
        "target": TARGET,
        "split": {
            "train_rows": len(X_train),
            "validation_rows": len(X_valid),
            "test_rows": len(X_test),
            "random_state": 42,
        },
        "test_metrics": {
            "sklearn_pipeline": sklearn_metrics,
            "onnx_runtime": onnx_metrics,
        },
        "prediction_parity": {
            "passed": parity_pass,
            "max_absolute_difference": max_abs_diff,
            "mean_absolute_difference": mean_abs_diff,
            "rtol": 1e-4,
            "atol": 1e-3,
        },
        "single_row_latency": {
            "sklearn_xgboost_estimator": sklearn_latency,
            "onnx_runtime": onnx_latency,
        },
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"\nEvaluation report saved to: {report_path}")

    if not parity_pass:
        raise AssertionError(
            "Sklearn and ONNX predictions failed the parity tolerance check."
        )

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate XGBoost sklearn and ONNX predictions."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--pipeline", type=Path, default=DEFAULT_PIPELINE_PATH)
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    args = parser.parse_args()

    evaluate(args.data, args.pipeline, args.onnx, args.report)


if __name__ == "__main__":
    main()