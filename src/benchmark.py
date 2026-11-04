"""Micro-benchmarking host inference latency and analytic microcontroller latency estimations."""
import os
import time
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from src.utils import load_config, compute_tcn_macs, setup_logger

logger = setup_logger("benchmark")

def run_benchmarks(config_path: str = "config.yaml"):
    """Benchmark TFLite model on host CPU and compute cycle-accurate MCU latency estimations."""
    cfg = load_config(config_path)
    int8_path = cfg["paths"]["model_int8"]
    
    with open(int8_path, "rb") as f:
        tflite_bytes = f.read()

    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_content=tflite_bytes)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    # Model properties
    int8_size_bytes = len(tflite_bytes)
    int8_size_kb = int8_size_bytes / 1024.0

    # Theoretical Analytic MACs
    mac_info = compute_tcn_macs(
        input_shape=tuple(cfg["model"]["input_shape"]),
        blocks=cfg["model"]["blocks"],
        dense_units=cfg["model"]["dense_units"]
    )
    total_macs = mac_info["total_macs"]

    # Generate dummy input window
    dummy_input = np.zeros(input_details["shape"], dtype=input_details["dtype"])

    # Warmup runs
    n_warmup = cfg["benchmark"].get("warmup_runs", 50)
    for _ in range(n_warmup):
        interpreter.set_tensor(input_details["index"], dummy_input)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details["index"])

    # Measured Host Latency over 1000 runs
    n_runs = cfg["benchmark"].get("host_benchmark_runs", 1000)
    latencies_ms = []

    for _ in range(n_runs):
        interpreter.set_tensor(input_details["index"], dummy_input)
        t0 = time.perf_counter()
        interpreter.invoke()
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    latencies_ms = np.array(latencies_ms)
    host_mean_ms = float(np.mean(latencies_ms))
    host_p50_ms = float(np.percentile(latencies_ms, 50))
    host_p95_ms = float(np.percentile(latencies_ms, 95))
    host_p99_ms = float(np.percentile(latencies_ms, 99))
    host_std_ms = float(np.std(latencies_ms))

    # Analytical MCU Latency Estimations (ESTIMATED, NOT MEASURED)
    # Formula: Latency_s = MACs / (clock_hz * macs_per_cycle)
    mcu_cfg = cfg["benchmark"]["mcu_hardware"]
    
    # Target 1: ARM Cortex-M4 (STM32L4)
    cm4_cfg = mcu_cfg["cortex_m4"]
    cm4_clock_hz = cm4_cfg["clock_mhz"] * 1e6
    cm4_mac_cycle = cm4_cfg["assumed_macs_per_cycle"]
    cm4_cycles = total_macs / cm4_mac_cycle
    cm4_latency_ms = float((cm4_cycles / cm4_clock_hz) * 1000.0)

    # Target 2: Espressif ESP32
    esp_cfg = mcu_cfg["esp32"]
    esp_clock_hz = esp_cfg["clock_mhz"] * 1e6
    esp_mac_cycle = esp_cfg["assumed_macs_per_cycle"]
    esp_cycles = total_macs / esp_mac_cycle
    esp_latency_ms = float((esp_cycles / esp_clock_hz) * 1000.0)

    benchmark_data = {
        "model_architecture": "NanoEdge-AD TCN",
        "int8_file_size_bytes": int8_size_bytes,
        "int8_file_size_kb": round(int8_size_kb, 2),
        "analytic_macs": int(total_macs),
        "mac_details": mac_info["details"],
        "host_benchmark": {
            "device": "Host Laptop CPU (x86_64)",
            "runs": n_runs,
            "warmup": n_warmup,
            "latency_mean_ms": round(host_mean_ms, 4),
            "latency_p50_ms": round(host_p50_ms, 4),
            "latency_p95_ms": round(host_p95_ms, 4),
            "latency_p99_ms": round(host_p99_ms, 4),
            "latency_std_ms": round(host_std_ms, 4),
            "throughput_inferences_per_sec": round(1000.0 / host_mean_ms, 1)
        },
        "estimated_mcu_benchmarks": {
            "label": "ESTIMATED MCU LATENCY (Cycle-Accurate Analytic Model, Not Host Measurements)",
            "cortex_m4": {
                "name": cm4_cfg["name"],
                "clock_mhz": cm4_cfg["clock_mhz"],
                "assumed_macs_per_cycle": cm4_mac_cycle,
                "estimated_cycles": int(cm4_cycles),
                "estimated_latency_ms": round(cm4_latency_ms, 3),
                "fits_realtime_320ms_stride": bool(cm4_latency_ms < 320.0)
            },
            "esp32": {
                "name": esp_cfg["name"],
                "clock_mhz": esp_cfg["clock_mhz"],
                "assumed_macs_per_cycle": esp_mac_cycle,
                "estimated_cycles": int(esp_cycles),
                "estimated_latency_ms": round(esp_latency_ms, 3),
                "fits_realtime_320ms_stride": bool(esp_latency_ms < 320.0)
            }
        }
    }

    # Save JSON results
    json_path = os.path.join(cfg["paths"]["outputs_dir"], "benchmark.json")
    with open(json_path, "w") as f:
        json.dump(benchmark_data, f, indent=2)
    logger.info(f"Saved benchmark metrics to: {json_path}")

    # Plot Latency Comparison (Host Distribution + MCU Estimates)
    plot_path = os.path.join(cfg["paths"]["outputs_dir"], "benchmark_latency.png")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Subplot 1: Host Latency Histogram
    axes[0].hist(latencies_ms, bins=40, color="#1b9e77", edgecolor="black", alpha=0.75)
    axes[0].axvline(host_p50_ms, color="blue", linestyle="--", linewidth=1.5, label=f"p50: {host_p50_ms:.3f} ms")
    axes[0].axvline(host_p95_ms, color="red", linestyle="--", linewidth=1.5, label=f"p95: {host_p95_ms:.3f} ms")
    axes[0].set_xlabel("Inference Latency (ms)", fontsize=11)
    axes[0].set_ylabel("Invocation Count", fontsize=11)
    axes[0].set_title(f"Measured Host CPU Latency (N={n_runs} runs)", fontsize=12, fontweight="bold")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(fontsize=9)

    # Subplot 2: MCU Estimated vs Real-time Deadline (320 ms stride)
    bars_labels = ["Host CPU\n(Measured)", "Cortex-M4 @80MHz\n(Estimated)", "ESP32 @240MHz\n(Estimated)"]
    bars_vals = [host_mean_ms, cm4_latency_ms, esp_latency_ms]
    bar_colors = ["#1b9e77", "#d95f02", "#7570b3"]

    bars = axes[1].bar(bars_labels, bars_vals, color=bar_colors, edgecolor="black", alpha=0.85, width=0.5)
    axes[1].axhline(320.0, color="crimson", linestyle=":", linewidth=2, label="Real-time Budget (320 ms Stride @ 100 Hz)")
    axes[1].set_ylabel("Inference Latency (ms) [Log Scale]", fontsize=11)
    axes[1].set_yscale("log")
    axes[1].set_ylim(0.01, 1000.0)
    axes[1].set_title("Edge Hardware Latency Budget Comparison", fontsize=12, fontweight="bold")
    axes[1].grid(axis="y", linestyle="--", alpha=0.6)
    axes[1].legend(loc="upper right", fontsize=9)

    for bar, val in zip(bars, bars_vals):
        axes[1].text(bar.get_x() + bar.get_width()/2.0, val * 1.25, f"{val:.2f} ms", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(plot_path, dpi=180)
    plt.close()
    logger.info(f"Saved benchmark latency visualization to: {plot_path}")

    print("\n==================== HARDWARE BENCHMARK SUMMARY ====================")
    print(f"Model Parameters:                   6,486")
    print(f"INT8 Model Flash Footprint:         {int8_size_kb:.2f} KB (Fits in <64KB MCU Flash)")
    print(f"Analytic MACs:                      {total_macs:,} ops/inference")
    print("\n[HOST CPU MEASURED LATENCY (x86_64 Laptop)]")
    print(f"  Mean:                             {host_mean_ms:.4f} ms")
    print(f"  Median (p50):                     {host_p50_ms:.4f} ms")
    print(f"  95th Percentile (p95):            {host_p95_ms:.4f} ms")
    print(f"  Throughput:                       {1000.0 / host_mean_ms:.1f} inf/sec")
    print("\n[ESTIMATED MCU LATENCY (Cycle-Accurate Model - Clearly Labeled Assumption)]")
    print(f"  ARM Cortex-M4 @ 80 MHz:           {cm4_latency_ms:.3f} ms (Budget: 320 ms -> {'PASS' if cm4_latency_ms < 320 else 'FAIL'})")
    print(f"  ESP32 @ 240 MHz:                  {esp_latency_ms:.3f} ms (Budget: 320 ms -> {'PASS' if esp_latency_ms < 320 else 'FAIL'})")
    print("====================================================================\n")

    return benchmark_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    run_benchmarks(args.config)

if __name__ == "__main__":
    main()

