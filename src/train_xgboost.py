
"""Train an XGBoost delivery-time model with MLflow tracking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import mlflow
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from src.data_loader import load_dataset
from src.preprocessing import TARGET, prepare_model_frame, make_preprocessor


def calculate_metrics(y_true, predictions):
    """Calculate regression metrics."""
    return {
        "mae": float(mean_absolute_error(y_true, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, predictions))),
        "r2": float(r2_score(y_true, predictions)),
    }


def train(
    data_path=None,
    output_dir="models",
    experiment_name="SwiftDrop-XGBoost",
):
    # Load and prepare the dataset.
    raw = load_dataset(data_path)
    frame = prepare_model_frame(raw)

    X = frame.drop(columns=[TARGET])
    y = frame[TARGET].astype(float)

    # 60% train, 20% validation, 20% test.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.4,
        random_state=42,
    )

    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=42,
    )

    print(f"Training rows:   {len(X_train)}")
    print(f"Validation rows: {len(X_valid)}")
    print(f"Test rows:       {len(X_test)}")

    model_params = {
        "n_estimators": 700,
        "max_depth": 7,
        "learning_rate": 0.04,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "objective": "reg:squarederror",
        "eval_metric": "mae",
        "random_state": 42,
        "n_jobs": -1,
    }

    model = XGBRegressor(**model_params)

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                make_preprocessor(X_train, scale_numeric=False),
            ),
            ("model", model),
        ]
    )

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        dataset_source_url = ("https://huggingface.co/datasets/allenborochin/zomato_delivery_EDA")

        # Track the dataset source as a tag.
        mlflow.set_tag("dataset_source_url", dataset_source_url)
        mlflow.set_tag("model_type", "XGBRegressor")
        mlflow.set_tag("framework", "scikit-learn Pipeline")

        # Track split configuration and all model hyperparameters.
        mlflow.log_params(model_params)
        mlflow.log_params(
            {
                "target": TARGET,
                "random_state": 42,
                "train_fraction": 0.6,
                "validation_fraction": 0.2,
                "test_fraction": 0.2,
            }
        )

        # Fit only on training data.
        pipeline.fit(X_train, y_train)

        # Use validation data for model assessment.
        valid_predictions = pipeline.predict(X_valid)
        validation_metrics = calculate_metrics(
            y_valid, valid_predictions
        )

        # Test metrics are reported separately, not used for tuning.
        test_predictions = pipeline.predict(X_test)
        test_metrics = calculate_metrics(y_test, test_predictions)

        mlflow.log_metric(
            "final_validation_mae",
            validation_metrics["mae"],
        )

        for name, value in validation_metrics.items():
            mlflow.log_metric(f"validation_{name}", value)

        for name, value in test_metrics.items():
            mlflow.log_metric(f"test_{name}", value)

        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

        pipeline_path = output / "xgboost_pipeline.joblib"
        metrics_path = output / "xgboost_metrics.json"

        joblib.dump(pipeline, pipeline_path)

        results = {
            "split": {
                "training_rows": len(X_train),
                "validation_rows": len(X_valid),
                "test_rows": len(X_test),
                "random_state": 42,
            },
            "validation_metrics": validation_metrics,
            "test_metrics": test_metrics,
            "mlflow_run_id": run.info.run_id,
        }

        metrics_path.write_text(
            json.dumps(results, indent=2),
            encoding="utf-8",
        )

        mlflow.log_artifact(str(metrics_path))

        print("\nValidation metrics:")
        print(json.dumps(validation_metrics, indent=2))

        print("\nTest metrics:")
        print(json.dumps(test_metrics, indent=2))

        print(f"\nPipeline saved to: {pipeline_path.resolve()}")
        print(f"Metrics saved to:  {metrics_path.resolve()}")
        print(f"MLflow run ID:     {run.info.run_id}")

        return pipeline, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default=None,
        help="Local source CSV path; omit to load the public dataset.",
    )
    parser.add_argument("--output-dir", default="models")
    args = parser.parse_args()

    train(data_path=args.data, output_dir=args.output_dir)
