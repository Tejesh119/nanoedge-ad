# NanoEdge-AD: Reproduction Requirements & System Specification

## 1. Environment & Hardware Specifications
- **Operating System**: `Windows 11 (AMD64)`
- **Python Version**: `3.13.5`
- **TensorFlow Version**: `2.21.0`
- **Seeds Evaluated**: `[42, 101, 777, 1234, 2024]`
- **Total Pipeline Execution Runtime**:
  - Full single-seed pipeline (`python run.py all`): ~2 minutes
  - Multi-seed evaluation (`python -m src.multiseed`): ~8-12 minutes
  - Diagnostic experiments suite (`python -m src.diagnostics`): ~6-8 minutes
  - Statistical aggregation & asset compilation (`python -m src.aggregate`): ~10 seconds

---

## 2. Reproduction Commands
```bash
# 1. Environment Activation
.venv\Scripts\activate  # On Linux/macOS: source .venv/bin/activate

# 2. Single-Seed Standard Pipeline
python run.py all
# Or: make all

# 3. Reviewer Multi-Seed Evaluation (5 Seeds: 42 101 777 1234 2024)
make multiseed
# Or: python -m src.multiseed --seeds 42 101 777 1234 2024

# 4. Reviewer Diagnostic Experiments Suite
make diagnostics
# Or: python -m src.diagnostics

# 5. Statistical Aggregation & Report Recompilation
make aggregate
# Or: python -m src.aggregate

# 6. Generate IEEE Paper Assets & LaTeX Tables
make paper-assets
# Or: python -m src.generate_paper_assets

# 7. Execute Pytest Unit Tests
python -m pytest tests/ -v
```

---

## 3. Pinned Dependencies (from `pip freeze`)
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
