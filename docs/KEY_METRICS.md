# NanoEdge-AD: Key Metrics & Empirical Results

> **Status**: Initialized (Phase 0). This document will be generated automatically by `python -m src.report` upon pipeline execution.

## Research Claims
- **C1**: A tiny quantized Temporal Convolutional Network (~8-12k params, int8) detects sensor anomalies with a lower false-positive rate than shallow models (decision tree) and comparable accuracy to larger fp32 models.
- **C2**: Dynamic feature (sensor) pruning cuts sensor reads substantially with small F1 loss.
- **C3**: The int8 model fits microcontroller-class budgets (size, params, MACs, host latency, ESTIMATED MCU latency).
- **C4**: The system is robust to noise, sensor dropout and drift.
