# SwiftDrop Delivery Prediction

An end-to-end machine-learning project that estimates food delivery duration from courier, order, weather, traffic, and location features. It compares an XGBoost regression baseline with TabNet and evaluates ONNX Runtime inference.

> **Before running:** Datasets and trained models are excluded from Git. Obtain the required artifacts from your configured artifact storage and place them in the paths listed below. Add a verified artifact download URL before publishing the repository.

## Dataset

- **Source:** [Zomato delivery EDA dataset on Hugging Face](https://huggingface.co/datasets/allenborochin/zomato_delivery_EDA)
- **Target:** `Time_taken (min)`
- **Expected local CSV:** `data/raw/zomato_cleaned.csv`

## Main components

- `src/data_loader.py` — dataset loading.
- `src/preprocessing.py` — cleaning, date/time feature engineering, train-fitted imputation/encoding, and leakage prevention.
- `src/train_xgboost.py` — XGBoost regression pipeline and MLflow tracking.
- `src/train_tabnet.py` — TabNet regressor and MLflow tracking.
- `src/export_xgboost_onnx.py` and `src/export_tabnet_onnx.py` — ONNX export.
- `src/evaluate_xgboost_onnx.py` and `src/evaluate_tabnet_onnx.py` — native-vs-ONNX evaluation and parity checks.
- `src/benchmark_xgboost_onnx.py` — XGBoost model-only batch inference benchmark.
- `app.py` — Streamlit prediction app with model selection and comparison.

## Setup (Windows PowerShell)

Use Python 3.11 for a consistent environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the app

The app expects these files in `models/`:

- `xgboost_pipeline.joblib`
- `tabnet_preprocessor.joblib`
- `tabnet.onnx`

The `xgboost.onnx` file is used by the XGBoost export/evaluation workflow; the current app uses the saved XGBoost sklearn pipeline. Download the required artifacts from the project's controlled artifact storage before launching:

```powershell
streamlit run app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`.

## Evaluation results

The models use the same deterministic 60/20/20 split (`random_state=42`): 23,378 training rows, 7,793 validation rows, and 7,793 test rows.

### Held-out test metrics

| Model | MAE (minutes) | RMSE (minutes) | R² |
|---|---:|---:|---:|
| XGBoost sklearn pipeline | 3.1252 | 3.9086 | 0.8231 |
| XGBoost ONNX Runtime | 3.1252 | 3.9086 | 0.8231 |
| TabNet native | 3.1764 | 3.9980 | 0.8149 |
| TabNet ONNX Runtime | 3.1764 | 3.9980 | 0.8149 |

XGBoost is the current accuracy leader on this held-out split.

### ONNX CPU batch benchmark

Model-only inference on 1,000 rows, averaged over 50 iterations. Preprocessing is excluded.

| Model | Runtime | Mean batch latency | P95 batch latency | Throughput |
|---|---|---:|---:|---:|
| XGBoost | Native estimator | 7.165 ms | 9.364 ms | 139,559 rows/s |
| XGBoost | ONNX Runtime CPU | 5.335 ms | 6.206 ms | 187,454 rows/s |
| TabNet | Native model | 23.083 ms | 29.295 ms | 43,322 rows/s |
| TabNet | ONNX Runtime CPU | 5.266 ms | 5.740 ms | 189,897 rows/s |

- XGBoost ONNX batch inference measured approximately **1.34×** the native estimator throughput.
- TabNet ONNX batch inference measured approximately **4.38×** the native model throughput.
- ONNX prediction-parity checks passed within the evaluation tolerances for both models.

These are model-only CPU measurements from the recorded environment. They exclude feature preparation, Streamlit overhead, networking, and deployment infrastructure latency. Results may vary by machine.

## Preprocessing and leakage controls

- Split data before fitting imputers and encoders.
- Preserve missing feature values until train-fitted preprocessing.
- Exclude IDs and `delivery_speed`, which is target-derived.
- Exclude pickup-time information because it may not be available when an order is placed.
- Parse order date/time and derive numeric and cyclic time features.
- Use `OneHotEncoder(handle_unknown="ignore")` for unseen categories.

The full-data no-null export is for inspection only and must not be used for validation/test metrics. See [PREPROCESSING_NOTES.md](PREPROCESSING_NOTES.md).

## Training and evaluation

Run from the repository root. Check each script's arguments before running; training/export commands may overwrite local artifacts.

```powershell
python -m src.train_xgboost
python -m src.train_tabnet
python -m src.export_xgboost_onnx
python -m src.export_tabnet_onnx
python -m src.evaluate_xgboost_onnx
python -m src.evaluate_tabnet_onnx
python -m src.benchmark_xgboost_onnx
```

To inspect local MLflow runs:

```powershell
mlflow ui
```

## Notebooks

- [01 — Data exploration](notebooks/01_data_exploration.ipynb)
- [02 — XGBoost baseline](notebooks/02_xgboost_baseline.ipynb)
- [03 — Neural network](notebooks/03_neural_network.ipynb)
- [04 — Neural-network ONNX evaluation](notebooks/04_neural_network_onnx_evaluation.ipynb)
- [05 — XGBoost ONNX evaluation](notebooks/05_xgboost_onnx_evaluation.ipynb)

## Consultant recommendation

Use **XGBoost as the initial primary model**: it achieved the better held-out test metrics in this experiment (MAE ≈ 3.13 minutes and R² ≈ 0.823). Its ONNX export closely matched the sklearn pipeline's predictions and improved model-only CPU batch inference throughput in the recorded benchmark.

Keep **TabNet as a challenger model**. Its ONNX export passed the recorded parity check and had a larger relative speedup over native TabNet, but its held-out error was slightly higher than XGBoost. Before production, validate on recent and geographically representative data, measure end-to-end latency including preprocessing, monitor deployed prediction error, and version models/artifacts.

## Git and artifact policy

Do not commit raw datasets, trained model binaries, `.joblib` files, ONNX files, local MLflow databases, virtual environments, or secrets. Store large artifacts in a controlled location (for example, Google Drive with suitable access controls) and add verified download instructions before making the repository public.
