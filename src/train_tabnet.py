
"""Train a TabNet delivery-time regression model with MLflow tracking."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import mlflow
import numpy as np
import torch
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.data_loader import load_dataset
from src.preprocessing import TARGET, prepare_model_frame, make_preprocessor


SEED = 42


def calculate_metrics(y_true, predictions):
    """Calculate regression metrics."""
    return {
        "mae": float(mean_absolute_error(y_true, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, predictions))),
        "r2": float(r2_score(y_true, predictions)),
    }


def train(data_path=None, output_dir="models"):
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    raw = load_dataset(data_path)
    frame = prepare_model_frame(raw)

    X = frame.drop(columns=[TARGET])
    y = frame[TARGET].astype(np.float32)

    # Match the XGBoost 60/20/20 split exactly.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=SEED
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=SEED
    )

    # Fit preprocessing only on the training split.
    preprocessor = make_preprocessor(X_train, scale_numeric=False)

    X_train_processed = preprocessor.fit_transform(X_train)
    X_valid_processed = preprocessor.transform(X_valid)
    X_test_processed = preprocessor.transform(X_test)

    # TabNet requires dense NumPy arrays.
    def to_dense_float32(values):
        if hasattr(values, "toarray"):
            values = values.toarray()
        return np.asarray(values, dtype=np.float32)

    X_train_processed = to_dense_float32(X_train_processed)
    X_valid_processed = to_dense_float32(X_valid_processed)
    X_test_processed = to_dense_float32(X_test_processed)

    y_train_np = y_train.to_numpy(dtype=np.float32).reshape(-1, 1)
    y_valid_np = y_valid.to_numpy(dtype=np.float32).reshape(-1, 1)
    y_test_np = y_test.to_numpy(dtype=np.float32).reshape(-1, 1)

    print(f"Training rows:   {len(X_train_processed)}")
    print(f"Validation rows: {len(X_valid_processed)}")
    print(f"Test rows:       {len(X_test_processed)}")
    print(f"Processed features: {X_train_processed.shape[1]}")

    params = {
        "seed": SEED,
        "max_epochs": 100,
        "patience": 10,
        "batch_size": 256,
        "virtual_batch_size": 128,
        "learning_rate": 0.02,
        "optimizer": "Adam",
        "mask_type": "sparsemax",
        "preprocessing": "existing sklearn preprocessor, numeric scaling disabled",
    }

    model = TabNetRegressor(
        seed=SEED,
        optimizer_fn=torch.optim.Adam,
        optimizer_params={"lr": params["learning_rate"]},
        mask_type=params["mask_type"],
        verbose=10,
        device_name="auto",
    )

    mlflow.set_experiment("SwiftDrop-TabNet")

    with mlflow.start_run() as run:
        mlflow.set_tag(
            "dataset_source_url",
            "https://huggingface.co/datasets/allenborochin/zomato_delivery_EDA",
        )
        mlflow.set_tag("model_type", "TabNetRegressor")
        mlflow.set_tag("framework", "pytorch-tabnet")
        mlflow.log_params(params)
        mlflow.log_params({
            "target": TARGET,
            "train_fraction": 0.6,
            "validation_fraction": 0.2,
            "test_fraction": 0.2,
            "random_state": SEED,
            "input_features": X_train_processed.shape[1],
        })

        model.fit(
            X_train=X_train_processed,
            y_train=y_train_np,
            eval_set=[
                (X_train_processed, y_train_np),
                (X_valid_processed, y_valid_np),
            ],
            eval_name=["train", "valid"],
            eval_metric=["mae"],
            max_epochs=params["max_epochs"],
            patience=params["patience"],
            batch_size=params["batch_size"],
            virtual_batch_size=params["virtual_batch_size"],
            num_workers=0,
            drop_last=False,
        )

        # Evaluate the selected model checkpoint.
        valid_predictions = model.predict(X_valid_processed).reshape(-1)
        test_predictions = model.predict(X_test_processed).reshape(-1)

        validation_metrics = calculate_metrics(
            y_valid_np.reshape(-1), valid_predictions
        )
        test_metrics = calculate_metrics(
            y_test_np.reshape(-1), test_predictions
        )

        mlflow.log_metric(
            "final_validation_mae", validation_metrics["mae"]
        )
        for name, value in validation_metrics.items():
            mlflow.log_metric(f"validation_{name}", value)
        for name, value in test_metrics.items():
            mlflow.log_metric(f"test_{name}", value)

        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

        preprocessor_path = output / "tabnet_preprocessor.joblib"
        model.save_model(str(output / "tabnet_model"))
        joblib.dump(preprocessor, preprocessor_path)

        results = {
            "model": "TabNetRegressor",
            "split": {
                "training_rows": len(X_train),
                "validation_rows": len(X_valid),
                "test_rows": len(X_test),
                "random_state": SEED,
            },
            "parameters": params,
            "validation_metrics": validation_metrics,
            "test_metrics": test_metrics,
            "preprocessor_path": str(preprocessor_path),
            "mlflow_run_id": run.info.run_id,
        }

        metrics_path = output / "tabnet_metrics.json"
        metrics_path.write_text(
            json.dumps(results, indent=2), encoding="utf-8"
        )
        mlflow.log_artifact(str(metrics_path))

        print("\nValidation metrics:")
        print(json.dumps(validation_metrics, indent=2))
        print("\nTest metrics:")
        print(json.dumps(test_metrics, indent=2))
        print(f"\nSaved model: {output / 'tabnet_model.zip'}")
        print(f"Saved preprocessor: {preprocessor_path}")
        print(f"Saved metrics: {metrics_path}")
        print(f"MLflow run ID: {run.info.run_id}")

        return model, results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None)
    parser.add_argument("--output-dir", default="models")
    args = parser.parse_args()

    train(data_path=args.data, output_dir=args.output_dir)
