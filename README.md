# SwiftDrop Delivery Prediction Pipeline

An end-to-end machine learning pipeline that predicts food-delivery time in minutes from order, courier, traffic, weather, and location-related features. The project compares **XGBoost regression** with **TabNet**, tracks experiments with MLflow, exports models to ONNX for inference benchmarking, and provides a Streamlit prediction app.

## Architecture

**[View the SwiftDrop Architecture Diagram (HTML)](.archify/architecture-swiftdrop-delivery-pipeline-20261019-103749/swiftdrop-delivery-pipeline.html)**

The architecture diagram covers:

- **Data pipeline:** Dataset loading, cleaning, feature engineering, and leakage prevention.
- **Model training:** Separate XGBoost and TabNet training paths with MLflow experiment tracking.
- **Model evaluation:** Held-out test metrics, ONNX export, and prediction-parity validation.
- **Inference benchmarking:** Native model inference compared with ONNX Runtime on CPU.
- **Prediction app:** Streamlit interface with separate XGBoost and TabNet inference paths.

The diagram passed Archify's automated validation checks. GitHub may display an HTML file as source rather than render it as an interactive webpage; host the file with GitHub Pages or another static host if you want the diagram to open as a live interactive page.

## Dataset

- **Dataset:** [Zomato delivery EDA dataset on Hugging Face](https://huggingface.co/datasets/allenborochin/zomato_delivery_EDA)
- **CSV source:** [zomato_cleaned.csv](https://huggingface.co/datasets/allenborochin/zomato_delivery_EDA/resolve/main/zomato_cleaned.csv)
- **Prediction target:** `Time_taken (min)`
- **Dataset size used in the project:** 38,964 rows

The data and generated model artifacts are not committed to this repository. Download or provide the dataset locally when required by a training command.

## Main components

| Path | Purpose |
|---|---|
| `src/data_loader.py` | Loads the local dataset or the public Hugging Face source. |
| `src/preprocessing.py` | Cleans data, parses time fields, engineers features, and prepares model inputs while preventing target leakage. |
| `src/train_xgboost.py` | Trains the XGBoost regression pipeline and logs experiment details to MLflow. |
| TabNet training code in `src/` | Trains the TabNet model using the shared preprocessing workflow. |
| `app.py` | Streamlit interface for delivery-time predictions. |
| `notebooks/` | Notebooks for exploratory analysis and model experimentation. |
| `PREPROCESSING_NOTES.md` | Additional preprocessing decisions and notes. |
| `.archify/architecture-swiftdrop-delivery-pipeline-20261019-103749/swiftdrop-delivery-pipeline.html` | Interactive architecture diagram generated with Archify. |

## Models and results

Both models were evaluated on a held-out test set.

| Model | MAE (minutes) | RMSE (minutes) | R² |
|---|---:|---:|---:|
| XGBoost | 3.1252 | 3.9086 | 0.8231 |
| TabNet | 3.1764 | 3.9980 | 0.8149 |

Lower MAE and RMSE are better; higher R² indicates that the model explains more of the target variance. In these experiments, XGBoost performed slightly better on the held-out test set.

## ONNX CPU inference benchmark

The following benchmark used a batch of 1,000 rows over 50 iterations. Preprocessing time was excluded, so these figures describe model inference only.

| Model | Native mean latency | ONNX mean latency | Native throughput | ONNX throughput | Speedup |
|---|---:|---:|---:|---:|---:|
| XGBoost | 7.165 ms | 5.335 ms | 139,559 rows/s | 187,454 rows/s | 1.343× |
| TabNet | 23.083 ms | 5.266 ms | 43,322 rows/s | 189,897 rows/s | 4.383× |

Prediction-parity checks passed for both ONNX exports. Benchmark values depend on the machine and runtime environment and should be interpreted as measurements from this project setup, not universal performance guarantees.

## Setup

Python 3.11 is recommended for the project's machine-learning dependencies.

### 1. Clone the repository

```bash
git clone https://github.com/abhimittal1062/swiftdrop-delivery-prediction.git
cd swiftdrop-delivery-prediction
```

### 2. Create and activate a virtual environment

**Windows PowerShell:**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux:**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Train and inspect experiments

Train the XGBoost model using a local CSV:

```bash
python -m src.train_xgboost --data data/raw/zomato_cleaned.csv
```

If `--data` is omitted, the data loader uses the configured public Hugging Face source.

Start the MLflow UI to inspect logged runs:

```bash
mlflow ui
```

Open the local URL printed by MLflow in your browser. Check the available training scripts under `src/` for the corresponding TabNet, ONNX export, evaluation, and benchmark workflows.

## Run the Streamlit app

After installing dependencies and ensuring the required model artifacts are available locally, run:

```bash
streamlit run app.py
```

The app needs the trained model/preprocessing artifacts expected by the project. Those generated artifacts are intentionally not stored in GitHub, so a fresh clone may require training the models or obtaining the artifacts from the project's artifact storage before the app can make predictions.

## Preprocessing and leakage prevention

- The raw dataset is used for model evaluation; the cleaned CSV is an inspection/sharing artifact.
- The data is split before fitting imputation and encoding, so preprocessing learns from training rows only.
- Identifier columns and target-derived leakage features are excluded from model inputs.
- Time fields are parsed and transformed into useful date/time features, including handling Excel-style fractions and values such as `24:05:00`.

See `PREPROCESSING_NOTES.md` for further implementation details.

## Artifact policy

Raw/processed datasets, fitted models, ONNX files, and other generated artifacts are excluded from Git to keep the repository lightweight. Store them in Google Drive or another artifact store and provide a stable download location if you want others to reproduce inference without retraining.

## Tech stack

- Python
- pandas and NumPy
- scikit-learn
- XGBoost
- PyTorch / TabNet
- MLflow
- ONNX and ONNX Runtime
- Streamlit

## Repository

GitHub: [abhimittal1062/swiftdrop-delivery-prediction](https://github.com/abhimittal1062/swiftdrop-delivery-prediction)

## Architecture Diagram

**[View the SwiftDrop Architecture Diagram (HTML)](.archify/architecture-swiftdrop-delivery-pipeline-20261019-103749/swiftdrop-delivery-pipeline.html)**
