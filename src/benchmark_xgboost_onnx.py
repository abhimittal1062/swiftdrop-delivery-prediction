from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort
from sklearn.model_selection import train_test_split

from src.data_loader import load_dataset
from src.preprocessing import TARGET, prepare_model_frame


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = ROOT_DIR / "data" / "raw" / "zomato_cleaned.csv"
DEFAULT_PIPELINE_PATH = ROOT_DIR / "models" / "xgboost_pipeline.joblib"
DEFAULT_ONNX_PATH = ROOT_DIR / "models" / "xgboost.onnx"
DEFAULT_REPORT_PATH = ROOT_DIR / "reports" / "xgboost_onnx_benchmark.json"


def benchmark_batch(predict_fn, X: np.ndarray, iterations: int = 50) -> dict:
    """Measure latency for predicting a batch of up to 1000 rows."""
    if len(X) == 0:
        raise ValueError("Cannot benchmark an empty dataset.")

    batch = X[: min(1000, len(X))]
    predict_fn(batch)  # warm-up

    latencies_ms = []
    for _ in range(iterations):
        start = time.perf_counter()
        predictions = predict_fn(batch)
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies_ms.append(elapsed_ms)

        if len(predictions) != len(batch):
            raise RuntimeError("Prediction count does not match batch size.")

    mean_ms = float(np.mean(latencies_ms))

    return {
        "rows_per_batch": len(batch),
        "iterations": iterations,
        "mean_ms": mean_ms,
        "median_ms": float(np.median(latencies_ms)),
        "p95_ms": float(np.percentile(latencies_ms, 95)),
        "rows_per_second": (
            float(len(batch) / (mean_ms / 1000)) if mean_ms > 0 else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark native XGBoost against ONNX Runtime."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--pipeline", type=Path, default=DEFAULT_PIPELINE_PATH)
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()

    for path in (args.pipeline, args.onnx):
        if not path.exists():
            raise FileNotFoundError(f"Required model file not found: {path}")

    raw_df = load_dataset(str(args.data) if args.data.exists() else None)
    df = prepare_model_frame(raw_df)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # Reproduce the same deterministic 60/20/20 split.
    _, X_temp, _, _ = train_test_split(
        X, y, test_size=0.4, random_state=42
    )
    _, X_test, _, _ = train_test_split(
        X_temp,
        df.loc[X_temp.index, TARGET],
        test_size=0.5,
        random_state=42,
    )

    pipeline = joblib.load(args.pipeline)
    preprocessor = pipeline.named_steps["preprocessor"]

    transformed = preprocessor.transform(X_test)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    X_test_transformed = np.asarray(transformed, dtype=np.float32)

    session = ort.InferenceSession(
        str(args.onnx),
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    def native_predict(batch):
        return pipeline.named_steps["model"].predict(batch)

    def onnx_predict(batch):
        result = session.run(
            [output_name],
            {input_name: np.asarray(batch, dtype=np.float32)},
        )[0]
        return np.asarray(result).reshape(-1)

    native_results = benchmark_batch(
        native_predict, X_test_transformed, args.iterations
    )
    onnx_results = benchmark_batch(
        onnx_predict, X_test_transformed, args.iterations
    )

    report = {
        "benchmark": "CPU model-only batch inference",
        "preprocessing_included": False,
        "requested_batch_size": 1000,
        "native_xgboost": native_results,
        "onnx_runtime": onnx_results,
        "onnx_speedup": (
            native_results["mean_ms"] / onnx_results["mean_ms"]
            if onnx_results["mean_ms"] > 0
            else None
        ),
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"\nBenchmark report saved to: {args.report}")


if __name__ == "__main__":
    main()