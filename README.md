# NanoEdge-AD: TinyML Real-Time Anomaly Detection for Multi-Sensor IoT Streams

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status: Experimental](https://img.shields.io/badge/Status-Research%20Prototype-green.svg)]()

> **Notice**: All sensor datasets used in this research repository are **synthetic** benchmark streams designed specifically to test edge anomaly detection under controlled noise, subtle drifts, and sensor degradation.

---

## 1. Design Summary & Architecture (15-Line Overview)
1. **Target**: Edge microcontrollers (e.g. ARM Cortex-M4, ESP32) monitoring 4-channel IoT sensor streams (100 Hz).
2. **Backbone**: Ultra-compact Temporal Convolutional Network (TCN) with ~8-12k parameters and full INT8 post-training quantization.
3. **Receptive Field**: 3 dilated residual blocks (d=1, 2, 4 with kernel size 3) capturing temporal dynamics over length-64 windows (640 ms).
4. **SensorGate Layer**: Trainable per-channel gate `x * sigmoid(4*g)` with L1 regularization to learn relative channel importance.
5. **Data Augmentation**: Training-time random channel-masking (1 to 3 channels) forcing the network to remain robust to missing sensor streams.
6. **Quantization**: Full-integer INT8 quantization using representative calibration windows, achieving sub-60 KB flash footprint.
7. **Runtime Pruning Policy**: Dynamic streaming scheduler skipping reads for low-importance, stable sensors to conserve battery energy.
8. **Dynamic Wakeup**: Instantaneous wake-up of all sensors upon detecting statistical deviation (z-score / variance jump) on active channels.
9. **Zero-Leakage Windowing**: Time-based chronological split (60/20/20 train/val/test); normalization statistics computed solely on training stream.
10. **Baselines**: Compared against Decision Tree (statistical features), Dense Autoencoder, Vanilla 1D-CNN, and Static Pruned CNN.
11. **Subtle Anomalies**: Five realistic degradation modes (spikes, gradual drift, stuck-at, noise bursts, dropouts) parameterized to prevent trivial 1.0 F1.
12. **Analytic MAC Counting**: Exact multiply-accumulate count verified via analytical layer formulations against hand-computed references.
13. **Latency Benchmarking**: Measured host wall-clock latency (mean/p50/p95) paired with clearly labeled, cycle-accurate MCU latency estimations.
14. **Energy Proxy**: Analytical calculation of energy saved via `E = N_reads * e_read + N_infer * e_infer` under documented assumptions.
15. **Reproducibility**: Fully deterministic execution via pinned YAML configurations, multi-seed validation, and comprehensive automated test suites.

---

## 2. Key Research Assumptions
- **Synthetic Data**: Sensor telemetry is generated synthetically (sinusoids + slow trends + Gaussian noise + correlated drift) to simulate physical IoT machinery.
- **Microcontroller Execution**: Host benchmarks are run on the local CPU; MCU inference latency is an analytical estimate based on clock speed and MAC throughput assumptions (STM32L4 Cortex-M4 @80MHz, 0.8 MAC/cycle; ESP32 @240MHz, 1.0 MAC/cycle).
- **Energy Proxy**: Energy consumption is modeled as an analytical proxy ($E = N_{\text{reads}} \times e_{\text{read}} + N_{\text{infer}} \times e_{\text{infer}}$) assuming standard $3.3\,\text{V}$ I2C/SPI sensor read overhead ($12\,\mu\text{J}$) and int8 inference ($45\,\mu\text{J}$).
- **Hardware Deployment Needed**: Real-world deployment on physical physical hardware (e.g., STM32 board or ESP32) with a physical dataset (e.g., CWRU bearing dataset) is identified as future experimental validation.

---

## 3. Reproduce
To reproduce the full pipeline from a clean checkout:
```bash
# Set up environment
python -m venv .venv
source .venv/bin/activate  # Or on Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Run complete pipeline
make all
# Alternatively on Windows systems without make:
python run.py all
```

See [RUN_REQUIREMENTS.md](docs/RUN_REQUIREMENTS.md) for detailed environment instructions and [KEY_METRICS.md](docs/KEY_METRICS.md) for empirical results.
