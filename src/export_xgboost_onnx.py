from __future__ import annotations

import argparse
from pathlib import Path

import onnxmltools
import onnx
from onnxmltools.convert.common.data_types import FloatTensorType
from xgboost import XGBRegressor


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_PIPELINE_PATH = ROOT_DIR / "models" / "xgboost_pipeline.joblib"
DEFAULT_OUTPUT_PATH = ROOT_DIR / "models" / "xgboost.onnx"


def export_model(
    pipeline_path: Path = DEFAULT_PIPELINE_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> None:
    import joblib

    if not pipeline_path.exists():
        raise FileNotFoundError(
            f"Trained pipeline not found: {pipeline_path}. "
            "Run src.train_xgboost first."
        )

    pipeline = joblib.load(pipeline_path)

    if "preprocessor" not in pipeline.named_steps or "model" not in pipeline.named_steps:
        raise ValueError(
            "Expected a sklearn Pipeline with 'preprocessor' and 'model' steps."
        )

    xgb_model = pipeline.named_steps["model"]

    if not isinstance(xgb_model, XGBRegressor):
        raise TypeError(
            f"Expected XGBRegressor, got {type(xgb_model).__name__}."
        )

    n_features = int(xgb_model.n_features_in_)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    initial_types = [
        ("float_input", FloatTensorType([None, n_features]))
    ]

    onnx_model = onnxmltools.convert_xgboost(
        xgb_model,
        initial_types=initial_types,
        target_opset=12,
    )

    onnx.checker.check_model(onnx_model)
    onnxmltools.utils.save_model(onnx_model, str(output_path))

    print(f"Exported ONNX model: {output_path}")
    print(f"Input feature count: {n_features}")
    print(f"Output file size: {output_path.stat().st_size / (1024 * 1024):.2f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export the trained XGBoost estimator to ONNX."
    )
    parser.add_argument(
        "--pipeline",
        type=Path,
        default=DEFAULT_PIPELINE_PATH,
        help="Path to the saved sklearn pipeline.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Destination path for the ONNX model.",
    )
    args = parser.parse_args()

    export_model(args.pipeline, args.output)


if __name__ == "__main__":
    main()