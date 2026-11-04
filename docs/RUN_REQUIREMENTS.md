# NanoEdge-AD: Reproduction Guide & System Requirements

This document provides exact system requirements, execution stages, and environment specifications to reproduce all empirical findings in NanoEdge-AD.

---

## 1. System & Hardware Specifications
- **Operating System**: Windows 11 (64bit)
- **Host CPU**: Single-socket x86_64 Laptop Processor
- **Host Memory**: >= 8 GB RAM recommended
- **Python Version**: 3.13.5 (Tested with Python 3.10+)
- **Primary Frameworks**: TensorFlow 2.21.0, NumPy, Pandas, Scikit-learn, Matplotlib

---

## 2. Setup & Installation
Create an isolated virtual environment and install the exact pinned dependencies:

```bash
# 1. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 2. Upgrade pip and install pinned requirements
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## 3. End-to-End Execution
To reproduce all experiments, models, tables, plots, and document artifacts from a clean checkout:

```bash
# Using Makefile
make all

# Cross-platform runner (Linux / macOS / Windows)
python run.py all
```

### Individual Execution Targets
Each stage is encapsulated as an independent CLI module:
- `python -m src.data_gen`: Generates synthetic 4-channel telemetry stream and sample plots.
- `python -m src.train`: Trains TCN and all baselines across 3 random seeds.
- `python -m src.quantize`: Converts primary model to full-integer INT8 TFLite model.
- `python -m src.prune_experiment`: Evaluates offline gate-ranked vs random pruning sweeps.
- `python -m src.runtime_pruning`: Executes chronological streaming simulation of PrunePolicy.
- `python -m src.benchmark`: Benchmarks host inference and computes MCU latency estimates.
- `python -m src.evaluate`: Generates confusion matrices, per-anomaly recall, and robustness curves.
- `python -m src.ablation`: Runs component ablation study across 5 system configurations.
- `python -m src.report`: Compiles `KEY_METRICS.md`, `RUN_REQUIREMENTS.md`, and `paper_draft.md`.
- `python -m pytest tests/ -v`: Runs comprehensive 8-item unit test suite.

---

## 4. Expected Output Files & Artifacts
Upon full pipeline execution, the following files are produced in `outputs/`:
| File | Approximate Size | Description |
|:---|:---|:---|
| `model_fp32.keras` | ~206 KB | Trained FP32 TCN model with SensorGate |
| `model_int8.tflite` | ~31.3 KB | Full INT8 post-training quantized TFLite binary |
| `gates.json` | ~150 B | Learned per-channel SensorGate activation weights |
| `test_windows.npz` | ~241 KB | 500 test windows with model predictions for hardware deployment |
| `metrics.json` | ~15 KB | Consolidated machine-readable single source of truth |
| `results.csv` / `results.md` | ~1 KB | Multi-seed comparison table across all models |
| `pruning_sweep.csv` | ~500 B | Gate-ranked vs Monte Carlo random pruning data |
| `runtime_simulation.csv` | ~400 B | Streaming simulation statistics and energy proxy |
| `benchmark.json` | ~2 KB | Host measured latency and MCU analytic estimates |
| `ablation.csv` | ~600 B | Component ablation study results |
| `paper_draft.md` | ~25 KB | Complete academic research paper draft |
| `sample_windows.png` | ~460 KB | Synthetic telemetry visualization across anomaly modes |
| `pruning_sweep.png` | ~180 KB | SensorGate weights and F1 vs reads saved trade-off |
| `runtime_simulation.png`| ~390 KB | Streaming timeline, instant wakeups, and read savings |
| `benchmark_latency.png` | ~110 KB | Host latency histogram and MCU latency budget comparison |
| `confusion_matrices.png`| ~55 KB | Confusion matrices for TCN and baselines |
| `per_anomaly_recall.png`| ~60 KB | Detection recall broken down by anomaly mode |
| `robustness_curves.png` | ~160 KB | F1 curves under noise, sensor dropout, and drift |
| `ablation_waterfall.png`| ~135 KB | Stepwise impact of Gate, INT8, and Pruning |

---

## 5. Pinned Dependencies (from `pip freeze`)
```text
absl-py==2.5.1
astunparse==1.6.3
certifi==2026.10.23
charset-normalizer==3.5.2
cloudpickle==3.1.2
colorama==0.4.6
contourpy==1.4.0
cycler==0.12.1
flatbuffers==25.12.19
fonttools==4.68.0
gast==0.7.0
google-pasta==0.2.0
grpcio==1.85.1
h5py==3.14.0
idna==3.20
iniconfig==2.3.1
joblib==1.6.0
keras==3.15.1
kiwisolver==1.5.1
libclang==18.1.1
markdown-it-py==4.2.0
matplotlib==3.11.2
mdurl==0.1.2
ml_dtypes==0.6.0
namex==0.1.0
narwhals==2.28.0
numpy==2.5.4
opt_einsum==3.4.0
optree==0.20.0
packaging==26.3
pandas==3.1.0
pillow==13.0.0
pluggy==1.7.0
protobuf==7.36.2
Pygments==2.21.0
pyparsing==3.3.3
pytest==9.2.0
python-dateutil==4.0.0
PyYAML==6.0.3
requests==2.34.2
```

---

## 6. Troubleshooting
- **No GPU Found Warning**: Native Windows running TensorFlow >= 2.11 defaults to CPU. This is expected and compliant with the laptop CPU constraint.
- **TFLite Interpreter Warning**: `tf.lite.Interpreter` deprecation warnings can be safely ignored as standard builtins are used.
