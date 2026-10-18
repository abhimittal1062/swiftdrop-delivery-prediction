
"""Evaluate native TabNet and ONNX Runtime on the same test split."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.data_loader import load_dataset
from src.preprocessing import TARGET, prepare_model_frame


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = ROOT_DIR / "data" / "raw" / "zomato_cleaned.csv"
DEFAULT_MODEL_PATH = ROOT_DIR / "models" / "tabnet_model.zip"
DEFAULT_PREPROCESSOR_PATH = ROOT_DIR / "models" / "tabnet_preprocessor.joblib"
DEFAULT_ONNX_PATH = ROOT_DIR / "models" / "tabnet.onnx"
DEFAULT_REPORT_PATH = ROOT_DIR / "reports" / "tabnet_onnx_evaluation.json"

SEED = 42


def calculate_metrics(y_true, predictions):
    return {
        "mae": float(mean_absolute_error(y_true, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, predictions))),
        "r2": float(r2_score(y_true, predictions)),
    }


def to_dense_float32(values):
    if hasattr(values, "toarray"):
        values = values.toarray()
    return np.asarray(values, dtype=np.float32)


def measure_latency_ms(predict_fn, X, warmup=20, iterations=100):
    """Measure single-row model inference latency; preprocessing excluded."""
    if len(X) == 0:
        raise ValueError("Cannot benchmark an empty dataset.")

    samples = X[:min(100, len(X))]

    for i in range(min(warmup, len(samples))):
        predict_fn(samples[i:i + 1])

    latencies = []
    for i in range(iterations):
        row = samples[i % len(samples): i % len(samples) + 1]
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


def measure_batch_latency_ms(
    predict_fn, X, batch_size=1000, warmup=5, iterations=50
):
    """Measure model-only latency for batches of up to 1,000 rows."""
    if len(X) == 0:
        raise ValueError("Cannot benchmark an empty dataset.")

    batch = X[:min(batch_size, len(X))]

    for _ in range(warmup):
        predict_fn(batch)

    latencies = []
    for _ in range(iterations):
        start = time.perf_counter()
        predict_fn(batch)
        latencies.append((time.perf_counter() - start) * 1000)

    mean_ms = float(np.mean(latencies))
    return {
        "measurement": "batch model inference; preprocessing excluded",
        "rows_per_batch": len(batch),
        "iterations": iterations,
        "mean_ms": mean_ms,
        "median_ms": float(np.median(latencies)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "throughput_rows_per_second": (
            float(len(batch) * 1000 / mean_ms) if mean_ms > 0 else None
        ),
    }


def evaluate(
    data_path=DEFAULT_DATA_PATH,
    model_path=DEFAULT_MODEL_PATH,
    preprocessor_path=DEFAULT_PREPROCESSOR_PATH,
    onnx_path=DEFAULT_ONNX_PATH,
    report_path=DEFAULT_REPORT_PATH,
):
    required_paths = (model_path, preprocessor_path, onnx_path)
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Required artifact not found: {path}")

    raw = load_dataset(str(data_path) if data_path.exists() else None)
    frame = prepare_model_frame(raw)

    X = frame.drop(columns=[TARGET])
    y = frame[TARGET]

    # Reproduce the exact deterministic 60/20/20 split used in training.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=SEED
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=SEED
    )

    preprocessor = joblib.load(preprocessor_path)
    X_test_transformed = to_dense_float32(
        preprocessor.transform(X_test)
    )
    y_test_np = y_test.to_numpy(dtype=np.float64)

    native_model = TabNetRegressor()
    native_model.load_model(str(model_path))

    def native_predict(batch):
        return np.asarray(
            native_model.predict(np.asarray(batch, dtype=np.float32))
        ).reshape(-1)

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

    # Test dynamic-batch ONNX inference on the full test set.
    native_predictions = native_predict(X_test_transformed).astype(np.float64)
    onnx_predictions = onnx_predict(X_test_transformed).astype(np.float64)

    native_metrics = calculate_metrics(y_test_np, native_predictions)
    onnx_metrics = calculate_metrics(y_test_np, onnx_predictions)

    differences = np.abs(native_predictions - onnx_predictions)
    parity_pass = bool(
        np.allclose(
            native_predictions,
            onnx_predictions,
            rtol=1e-4,
            atol=1e-3,
        )
    )

    native_single_latency = measure_latency_ms(
        native_predict, X_test_transformed
    )
    onnx_single_latency = measure_latency_ms(
        onnx_predict, X_test_transformed
    )

    native_batch_latency = measure_batch_latency_ms(
        native_predict, X_test_transformed
    )
    onnx_batch_latency = measure_batch_latency_ms(
        onnx_predict, X_test_transformed
    )

    native_batch_mean = native_batch_latency["mean_ms"]
    onnx_batch_mean = onnx_batch_latency["mean_ms"]

    report = {
        "dataset": str(data_path),
        "target": TARGET,
        "split": {
            "train_rows": len(X_train),
            "validation_rows": len(X_valid),
            "test_rows": len(X_test),
            "random_state": SEED,
        },
        "test_metrics": {
            "native_tabnet": native_metrics,
            "onnx_runtime": onnx_metrics,
        },
        "prediction_parity": {
            "passed": parity_pass,
            "max_absolute_difference": float(np.max(differences)),
            "mean_absolute_difference": float(np.mean(differences)),
            "rtol": 1e-4,
            "atol": 1e-3,
        },
        "single_row_latency": {
            "native_tabnet": native_single_latency,
            "onnx_runtime": onnx_single_latency,
        },
        "batch_latency": {
            "native_tabnet": native_batch_latency,
            "onnx_runtime": onnx_batch_latency,
            "onnx_speedup": (
                float(native_batch_mean / onnx_batch_mean)
                if onnx_batch_mean > 0 else None
            ),
            "preprocessing_included": False,
        },
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"\nEvaluation report saved to: {report_path}")

    if not parity_pass:
        raise AssertionError(
            "Native TabNet and ONNX predictions failed the parity check."
        )

    print("Native TabNet vs ONNX parity check PASSED.")
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate native TabNet and ONNX Runtime."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--preprocessor", type=Path, default=DEFAULT_PREPROCESSOR_PATH
    )
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    args = parser.parse_args()

    evaluate(
        data_path=args.data,
        model_path=args.model,
        preprocessor_path=args.preprocessor,
        onnx_path=args.onnx,
        report_path=args.report,
    )


if __name__ == "__main__":
    main()
