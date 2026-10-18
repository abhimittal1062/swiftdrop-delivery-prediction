
"""Export the trained TabNet regressor to ONNX and verify parity."""

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from pytorch_tabnet.tab_model import TabNetRegressor


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "tabnet_model.zip"
ONNX_PATH = ROOT / "models" / "tabnet.onnx"


class TabNetPredictionWrapper(nn.Module):
    """Return predictions only, excluding TabNet's auxiliary loss."""

    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, x):
        predictions, _ = self.network(x)
        return predictions


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")

    model = TabNetRegressor()
    model.load_model(str(MODEL_PATH))

    model.network.eval()
    wrapper = TabNetPredictionWrapper(model.network).eval()

    # The current preprocessing pipeline produces 46 features.
    dummy_input = torch.randn(4, 46, dtype=torch.float32)

    with torch.inference_mode():
        native_predictions = wrapper(dummy_input).cpu().numpy()

    print("Native output shape:", native_predictions.shape)

    torch.onnx.export(
        wrapper,
        (dummy_input,),
        str(ONNX_PATH),
        input_names=["features"],
        output_names=["predictions"],
        dynamic_axes={
            "features": {0: "batch_size"},
            "predictions": {0: "batch_size"},
        },
        opset_version=17,
        dynamo=False,
    )

    # Check the exported model's ONNX structure.
    onnx_model = onnx.load(str(ONNX_PATH))
    onnx.checker.check_model(onnx_model)

    # Verify predictions using ONNX Runtime CPU.
    session = ort.InferenceSession(
        str(ONNX_PATH),
        providers=["CPUExecutionProvider"],
    )

    onnx_predictions = session.run(
        ["predictions"],
        {"features": dummy_input.numpy()},
    )[0]

    max_difference = float(
        np.max(np.abs(native_predictions - onnx_predictions))
    )
    mean_difference = float(
        np.mean(np.abs(native_predictions - onnx_predictions))
    )

    print("ONNX output shape:", onnx_predictions.shape)
    print(f"Maximum absolute difference: {max_difference:.8f}")
    print(f"Mean absolute difference:    {mean_difference:.8f}")

    if not np.allclose(
        native_predictions,
        onnx_predictions,
        rtol=1e-4,
        atol=1e-3,
    ):
        raise RuntimeError("ONNX parity check FAILED.")

    print("ONNX parity check PASSED.")
    print(f"ONNX model saved to: {ONNX_PATH}")
    print(f"File size: {ONNX_PATH.stat().st_size / (1024 * 1024):.2f} MB")


if __name__ == "__main__":
    main()
