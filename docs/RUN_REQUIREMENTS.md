# NanoEdge-AD: Run Requirements and Reproduction Guide

## 1. System Environment
- **Operating System**: Windows 11 / Windows 10 (AMD64)
- **Python Version**: Python 3.10+ (tested on Python 3.13.5)
- **Framework**: TensorFlow 2.21.0, NumPy, Pandas, Scikit-learn, Matplotlib, PyYAML, PyTest
- **Hardware Profile**: Laptop CPU (Single-socket x86_64)

## 2. Setup Commands
```powershell
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install pinned dependencies
python -m pip install -r requirements.txt
```

## 3. Pipeline Execution
All stages can be executed individually via CLI modules:
- Data Generation: `python -m src.data_gen`
- Baseline & TCN Training: `python -m src.train`
- INT8 Quantization: `python -m src.quantize`
- Offline Pruning Sweeps: `python -m src.prune_experiment`
- Runtime Streaming Simulation: `python -m src.runtime_pruning`
- Micro-benchmarking: `python -m src.benchmark`
- Multi-Metric Evaluation: `python -m src.evaluate`
- Ablation Studies: `python -m src.ablation`
- Report & Summary Generation: `python -m src.report`
- Unit Testing: `pytest tests/ -v`

Or run the complete suite:
```powershell
python run.py all
# or if make is installed:
make all
```

## 4. Dependencies
Will be updated automatically with exact pinned versions from `pip freeze`.
