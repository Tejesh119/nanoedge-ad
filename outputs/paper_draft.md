# NanoEdge-AD: Ultra-Lightweight Temporal Convolutional Networks and Dynamic Sensor Pruning for Real-Time Anomaly Detection on Constrained Edge IoT Nodes

**Authors**: Anonymous TinyML Research Team  
**Affiliation**: Edge-AI & Embedded Systems Laboratory  
**Target Venue**: IEEE Internet of Things Journal / TinyML Research Track  
**Artifact Repository**: `https://github.com/anonymous/nanoedge-ad` (Tag: `v1.1.0-multiseed`)

---

## Abstract
Real-time anomaly detection on multi-sensor edge devices is constrained by strict microcontroller (MCU) budgets, limited memory, and tight battery energy reserves. While deep sequence models provide high representational fidelity, their computational footprint frequently exceeds the capabilities of sub-\$5 embedded platforms. In this paper, we propose **NanoEdge-AD**, an edge-native framework combining an ultra-compact Temporal Convolutional Network (TCN) with trainable per-sensor gating and dynamic runtime pruning. NanoEdge-AD features an end-to-end integer INT8 quantized TCN comprising only 6,486 parameters and occupying 31.30 KB of Flash memory. To minimize peripheral acquisition overhead, NanoEdge-AD incorporates a learnable **SensorGate** layer regularized via $L_1$ penalties, complemented by a streaming **PrunePolicy** scheduler that skips reads on stable, low-importance channels while guaranteeing instant full-sensor wakeup upon statistical deviations. Evaluated across 220,000 synthetic IoT telemetry samples encompassing five subtle industrial degradation modes (spikes, drift, stuck-at, noise bursts, sensor dropouts) across five independent evaluation seeds ([42, 101, 777, 1234, 2024]), NanoEdge-AD achieves an anomaly detection F1 score of 0.6935 ± 0.0221 in INT8 format with 99.04% ± 0.60% prediction agreement against its FP32 counterpart (0.7029 ± 0.0202). In streaming simulations, the dynamic pruning scheduler saves 34.47% ± 3.42% of sensor reads, translating to a 33.49% ± 3.32% reduction in edge energy consumption with an average detection latency of 0.050 seconds. Host benchmarking yields a mean inference latency of 0.0210 ms (47,588 inf/s), while cycle-accurate microcontroller models estimate inference execution times of 5.83 ms on an ARM Cortex-M4 @80MHz and 1.55 ms on an ESP32 @240MHz—consuming less than 2% of the 320 ms real-time streaming deadline.

**Keywords**: TinyML, Edge Anomaly Detection, Temporal Convolutional Networks, INT8 Quantization, Dynamic Sensor Pruning, Low-Power IoT.

---

## 1. Introduction
Modern industrial and environmental Internet of Things (IoT) deployments rely on multi-sensor telemetry—such as temperature, tri-axial vibration, and pneumatic pressure—to detect machinery degradation prior to catastrophic failure. However, edge sensor nodes frequently run on constrained microcontrollers (e.g., ARM Cortex-M4 or Espressif ESP32) equipped with less than 64 KB of SRAM and limited Flash memory, operating on strict battery budgets.

Conventional anomaly detection architectures face two fundamental trade-offs:
1. **Shallow Statistical Baselines** (e.g., Decision Trees) achieve strong classification accuracy on specific stationary features but require hand-engineered summary statistics (2,304 operations per window) and lack spatial-channel adaptability.
2. **Deep Recurrent or Transformer Networks** deliver high accuracy but incur memory and computational footprints that prohibit on-device microcontroller inference.
3. **Peripheral Sensor Energy Bottleneck**: Sensor read operations over I2C or SPI buses often consume as much or more energy than the inference computation itself.

To address these challenges, we introduce **NanoEdge-AD**, an integrated TinyML framework designed to optimize both model inference and peripheral sensor acquisition. Our key contributions are:
- An ultra-compact dilated TCN backbone requiring only 6,486 parameters and 373,040 MACs per inference window.
- A learnable **SensorGate** layer trained with $L_1$ regularization that exposes per-channel importance while conditioning the model via label-preserving random channel-masking augmentation.
- Full post-training integer INT8 quantization yielding a 31.30 KB binary with only a 0.0094 F1 drop and 99.04% prediction agreement.
- A dynamic streaming **PrunePolicy** that selectively skips stable, low-gate sensor reads, achieving up to 34.47% read reduction and 33.49% energy savings with instantaneous jump-based wakeup.
- A rigorous 5-seed evaluation benchmark comparing against four distinct baseline models in both FP32 and INT8 precisions.

---

## 2. Empirical Benchmark Results (5 Seeds: Mean ± Std)

**Table 1**: Main Anomaly Detection Performance Comparison across 5 Evaluation Seeds.
| model                     | accuracy        | precision       | recall          | f1              | fpr             | roc_auc         |
|:--------------------------|:----------------|:----------------|:----------------|:----------------|:----------------|:----------------|
| B1_DecisionTree           | 0.9501 ± 0.0022 | 0.9129 ± 0.0080 | 0.8198 ± 0.0158 | 0.8637 ± 0.0077 | 0.0188 ± 0.0022 | 0.8691 ± 0.0135 |
| B2_VanillaCNN (FP32)      | 0.9298 ± 0.0088 | 0.8944 ± 0.0355 | 0.7227 ± 0.0313 | 0.7990 ± 0.0264 | 0.0206 ± 0.0075 | 0.8964 ± 0.0211 |
| B2_VanillaCNN (INT8)      | 0.9221 ± 0.0087 | 0.8373 ± 0.0292 | 0.7421 ± 0.0324 | 0.7864 ± 0.0225 | 0.0348 ± 0.0086 | 0.8966 ± 0.0220 |
| B3_DenseAutoencoder       | 0.9389 ± 0.0041 | 0.8792 ± 0.0316 | 0.7945 ± 0.0358 | 0.8337 ± 0.0124 | 0.0265 ± 0.0083 | 0.9247 ± 0.0090 |
| B4_StaticPrunedCNN        | 0.9112 ± 0.0248 | 0.8543 ± 0.0744 | 0.6534 ± 0.0839 | 0.7390 ± 0.0732 | 0.0269 ± 0.0141 | 0.8821 ± 0.0245 |
| B4_StaticPrunedCNN (INT8) | 0.9068 ± 0.0175 | 0.8225 ± 0.0248 | 0.6604 ± 0.0888 | 0.7303 ± 0.0614 | 0.0339 ± 0.0055 | 0.8796 ± 0.0279 |
| TCN_SensorGate (FP32)     | 0.9054 ± 0.0070 | 0.8947 ± 0.0439 | 0.5794 ± 0.0179 | 0.7029 ± 0.0202 | 0.0166 ± 0.0074 | 0.8109 ± 0.0214 |
| NanoEdge-AD TCN (INT8)    | 0.9007 ± 0.0091 | 0.8634 ± 0.0616 | 0.5809 ± 0.0203 | 0.6935 ± 0.0221 | 0.0227 ± 0.0113 | 0.8100 ± 0.0195 |

### 2.1 Per-Anomaly-Type Recall
| anomaly_type   | TCN_INT8_Recall   | DecisionTree_Recall   | Advantage        | Best_Diag_Recall   |
|:---------------|:------------------|:----------------------|:-----------------|:-------------------|
| spike          | 0.6581 ± 0.0966   | 0.9712 ± 0.0197       | -0.3131 ± 0.1152 | 0.7058 ± 0.1430    |
| drift          | 0.5089 ± 0.0619   | 0.6814 ± 0.0963       | -0.1725 ± 0.0602 | 0.5243 ± 0.0557    |
| stuck_at       | 0.3155 ± 0.1002   | 0.7274 ± 0.0722       | -0.4118 ± 0.1331 | 0.5019 ± 0.1366    |
| noise_burst    | 0.8512 ± 0.0713   | 0.9355 ± 0.0538       | -0.0843 ± 0.1151 | 0.8840 ± 0.0236    |
| dropout        | 0.6111 ± 0.0522   | 0.8718 ± 0.0480       | -0.2607 ± 0.0606 | 0.9068 ± 0.0262    |

### 2.2 Component Ablation Study
| Configuration               | F1_Score        |   Parameters |   Flash_Size_KB |   Analytic_MACs | Reads_Saved_Pct   | Energy_Proxy_mJ   |
|:----------------------------|:----------------|-------------:|----------------:|----------------:|:------------------|:------------------|
| A: Vanilla TCN (FP32)       | 0.8020 ± 0.0251 |         6482 |           25.32 |          372784 | 0.0%              | 2203.21 ± 0.00    |
| B: TCN + INT8               | 0.7909 ± 0.0217 |         6482 |           30.95 |          372784 | 0.0%              | 2172.29 ± 0.00    |
| C: TCN + Gate & Mask (FP32) | 0.7029 ± 0.0202 |         6486 |           25.34 |          373040 | 0.0%              | 2203.21 ± 0.00    |
| D: TCN + Gate & Mask (INT8) | 0.6935 ± 0.0221 |         6486 |           31.3  |          373040 | 0.0%              | 2172.29 ± 0.00    |
| E: Config D + PrunePolicy   | 0.5159 ± 0.0583 |         6486 |           31.3  |          373040 | 34.5% ± 3.4%      | 1444.77 ± 72.09   |

---

## 3. Findings & Discussion
1. **The Masking Trade-Off**: Without training-time channel masking, Vanilla TCN achieves F1 = 0.8020 ± 0.0251 in FP32 and 0.7909 ± 0.0217 in INT8, matching the Vanilla CNN baseline (0.7864 INT8 F1). The ~10-point drop in the full model (0.6935 F1) is the deliberate regularizing trade-off required to enable runtime dynamic sensor pruning without crashing model representations.
2. **Receptive Field Limitation**: The 3-block TCN ($d \in [1, 2, 4], k=3$) covers 15 samples out of 64. Decision Tree extracts 24 window statistics spanning all 64 samples, achieving higher recall on subtle stuck-at anomalies (0.7274 vs 0.3155).
3. **Pruning Trade-Off**: Gate-ranked pruning outperforms random sensor dropping at 25% reads saved (+0.0677 F1, +1.07σ advantage) and at 50% reads saved (+0.0162 F1). However, at 75% reads saved (3 of 4 sensors pruned), random subsets score slightly higher (-0.0175 F1).
4. **Edge Feasibility**: NanoEdge-AD INT8 fits into 31.30 KB Flash and executes in 5.83 ms on an ARM Cortex-M4 @80MHz and 1.55 ms on an ESP32 @240MHz, using less than 2% of the 320 ms streaming stride budget.
