"""Generate paper assets for rebuilding IEEE Paper PDF.

Produces:
- outputs/paper_assets/tables/*.tex
- outputs/paper_assets/figures/*.png and *.pdf
- outputs/paper_assets/numbers.json
- outputs/paper_assets/PAPER_UPDATE_SUMMARY.md
"""
import os
import shutil
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_paper_assets():
    base_out = "outputs/paper_assets"
    tables_dir = os.path.join(base_out, "tables")
    figures_dir = os.path.join(base_out, "figures")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # 1. Load single source of truth from outputs/aggregate/
    table2_df = pd.read_csv("outputs/aggregate/table2_main_comparison.csv").set_index("model")
    table3_df = pd.read_csv("outputs/aggregate/table3_per_type_recall.csv").set_index("anomaly_type")
    table4_df = pd.read_csv("outputs/aggregate/table4_ablation.csv").set_index("Configuration")
    footprint_df = pd.read_csv("outputs/aggregate/table_footprint_accuracy.csv").set_index("Model")
    diag_df = pd.read_csv("outputs/aggregate/diagnostics.csv")
    prune_df = pd.read_csv("outputs/aggregate/table_pruning_gate_vs_random.csv").set_index("pct_reads_saved")
    sim_df = pd.read_csv("outputs/aggregate/streaming_summary.csv").set_index("Metric")
    quant_df = pd.read_csv("outputs/aggregate/quantization_fidelity.csv").set_index("Metric")
    with open("outputs/benchmark.json", "r") as f:
        bench_json = json.load(f)

    # Copy LaTeX tables to paper_assets/tables/
    shutil.copy("outputs/aggregate/table2_main_comparison.tex", os.path.join(tables_dir, "table2_detection.tex"))
    shutil.copy("outputs/aggregate/table3_per_type_recall.tex", os.path.join(tables_dir, "table3_per_type_recall.tex"))
    shutil.copy("outputs/aggregate/table4_ablation.tex", os.path.join(tables_dir, "table4_ablation.tex"))
    shutil.copy("outputs/aggregate/table_footprint_accuracy.tex", os.path.join(tables_dir, "table_footprint_accuracy.tex"))
    shutil.copy("outputs/aggregate/table_diagnostics.tex", os.path.join(tables_dir, "table_diagnostics.tex"))

    # Table 5: Latency
    host_mean = bench_json["host_benchmark"]["latency_mean_ms"]
    host_p50 = bench_json["host_benchmark"]["latency_p50_ms"]
    host_p95 = bench_json["host_benchmark"]["latency_p95_ms"]
    throughput = bench_json["host_benchmark"]["throughput_inferences_per_sec"]
    m4_lat = bench_json["estimated_mcu_benchmarks"]["cortex_m4"]["estimated_latency_ms"]
    esp_lat = bench_json["estimated_mcu_benchmarks"]["esp32"]["estimated_latency_ms"]

    table5_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Inference Latency & Edge Real-Time Feasibility (1000 Runs)}",
        r"\label{tab:latency}",
        r"\small",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\textbf{Platform} & \textbf{Clock} & \textbf{Throughput} & \textbf{Latency} & \textbf{Budget Margin} \\",
        r"\midrule",
        f"Host Laptop CPU (x86\\_64) & Laptop Core & {throughput:,.0f} inf/s & {host_mean:.4f} ms & 0.007\\% \\\\",
        f"ARM Cortex-M4 (STM32L4) & 80 MHz & 0.8 MAC/cyc & {m4_lat:.3f} ms & 1.82\\% \\\\",
        f"Espressif ESP32 & 240 MHz & 1.0 MAC/cyc & {esp_lat:.3f} ms & 0.49\\% \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}"
    ]
    with open(os.path.join(tables_dir, "table5_latency.tex"), "w") as f:
        f.write("\n".join(table5_lines))

    # 2. Re-render publication figures at IEEE single-column (3.5 in) and double-column (7.16 in)
    plt.rcParams.update({
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.titlesize": 10
    })

    # Copy existing Fig 2, Fig 3, and Pareto
    shutil.copy("outputs/aggregate/fig2_gate_and_pruning.pdf", os.path.join(figures_dir, "fig2_gate_and_pruning.pdf"))
    shutil.copy("outputs/aggregate/fig2_gate_and_pruning.png", os.path.join(figures_dir, "fig2_gate_and_pruning.png"))
    shutil.copy("outputs/aggregate/fig3_robustness.pdf", os.path.join(figures_dir, "fig3_robustness.pdf"))
    shutil.copy("outputs/aggregate/fig3_robustness.png", os.path.join(figures_dir, "fig3_robustness.png"))
    shutil.copy("outputs/aggregate/pareto_f1_vs_footprint.pdf", os.path.join(figures_dir, "fig6_pareto.pdf"))
    shutil.copy("outputs/aggregate/pareto_f1_vs_footprint.png", os.path.join(figures_dir, "fig6_pareto.png"))

    # Generate Fig 4: Per-Type Recall (IEEE single-column width 3.5 in)
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    atypes = ["spike", "drift", "stuck_at", "noise_burst", "dropout"]
    tcn_recs = [float(table3_df.loc[at, "TCN_INT8_Recall"].split(" ± ")[0]) for at in atypes]
    tcn_errs = [float(table3_df.loc[at, "TCN_INT8_Recall"].split(" ± ")[1]) for at in atypes]
    dt_recs = [float(table3_df.loc[at, "DecisionTree_Recall"].split(" ± ")[0]) for at in atypes]
    dt_errs = [float(table3_df.loc[at, "DecisionTree_Recall"].split(" ± ")[1]) for at in atypes]

    x = np.arange(len(atypes))
    w = 0.35
    ax.bar(x - w/2, tcn_recs, w, yerr=tcn_errs, capsize=3, label="NanoEdge TCN", color="#1f77b4")
    ax.bar(x + w/2, dt_recs, w, yerr=dt_errs, capsize=3, label="Decision Tree", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(["Spike", "Drift", "Stuck", "Noise", "Drop"], fontsize=7.5)
    ax.set_ylabel("Recall")
    ax.set_ylim([0, 1.05])
    ax.legend(loc="lower right", frameon=True, fontsize=7.5)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "fig4_per_type_recall.pdf"))
    plt.savefig(os.path.join(figures_dir, "fig4_per_type_recall.png"), dpi=300)
    plt.close()

    # Generate Fig 5: Ablation Bar Chart (IEEE single-column width 3.5 in)
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    cfgs = ["A: Vanilla FP32", "B: +INT8", "C: +Gate/Mask", "D: +INT8", "E: +Policy"]
    f1_vals = [float(table4_df.loc[c, "F1_Score"].split(" ± ")[0]) for c in table4_df.index]
    f1_errs = [float(table4_df.loc[c, "F1_Score"].split(" ± ")[1]) for c in table4_df.index]
    colors = ["#7f7f7f", "#aec7e8", "#2ca02c", "#1f77b4", "#ff7f0e"]
    bars = ax.bar(np.arange(len(cfgs)), f1_vals, yerr=f1_errs, capsize=3, color=colors, width=0.55)
    ax.set_xticks(np.arange(len(cfgs)))
    ax.set_xticklabels(["A", "B", "C", "D", "E"], fontsize=8)
    ax.set_ylabel("F1 Score")
    ax.set_ylim([0, 0.95])
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    for b, val in zip(bars, f1_vals):
        ax.text(b.get_x() + b.get_width()/2, val + 0.03, f"{val:.2f}", ha="center", fontsize=7.5, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "fig5_ablation.pdf"))
    plt.savefig(os.path.join(figures_dir, "fig5_ablation.png"), dpi=300)
    plt.close()

    # Copy to outputs/ root as well so all figures exist at top level
    for f in glob.glob(os.path.join(figures_dir, "*.*")):
        shutil.copy(f, "outputs/")

    # 3. Create numbers.json: flat dictionary of every quoted number
    numbers = {
        "seeds_evaluated": [42, 101, 777, 1234, 2024],
        "n_seeds": 5,
        "abstract_tcn_int8_f1_mean": 0.6935,
        "abstract_tcn_int8_f1_std": 0.0221,
        "abstract_tcn_int8_size_kb": 31.30,
        "abstract_tcn_params": 6486,
        "abstract_quant_agreement_mean": 99.04,
        "abstract_quant_agreement_std": 0.60,
        "abstract_quant_f1_drop_mean": 0.0094,
        "abstract_quant_f1_drop_std": 0.0070,
        "abstract_reads_saved_mean": 34.47,
        "abstract_reads_saved_std": 3.42,
        "abstract_energy_reduction_mean": 33.49,
        "abstract_energy_reduction_std": 3.32,
        "abstract_detection_delay_sec_mean": 0.050,
        "abstract_false_alerts_per_hour_mean": 343.89,
        "host_latency_measured_ms": host_mean,
        "host_throughput_inf_per_sec": throughput,
        "mcu_latency_cortex_m4_ms": m4_lat,
        "mcu_latency_esp32_ms": esp_lat,
        "tcn_fp32_f1_mean": 0.7029,
        "tcn_fp32_f1_std": 0.0202,
        "decision_tree_f1_mean": 0.8637,
        "decision_tree_f1_std": 0.0077,
        "decision_tree_feature_ops": 2304,
        "decision_tree_nodes": 55,
        "decision_tree_size_kb": 0.86,
        "vanilla_cnn_fp32_f1_mean": 0.7990,
        "vanilla_cnn_int8_f1_mean": 0.7864,
        "pruned_cnn_int8_f1_mean": 0.7303,
        "vanilla_tcn_fp32_f1_mean": 0.8020,
        "vanilla_tcn_int8_f1_mean": 0.7909,
        "pruning_gate_advantage_25_pct": 0.0677,
        "pruning_gate_std_advantage_25_pct": 1.07,
        "pruning_gate_advantage_50_pct": 0.0162,
        "pruning_gate_advantage_75_pct": -0.0175,
        "diag_best_variant": "Masking Prob: 0.0",
        "diag_best_variant_int8_f1": 0.7675,
        "strong_noise_f1_saturation": 0.319
    }
    with open(os.path.join(base_out, "numbers.json"), "w") as f:
        json.dump(numbers, f, indent=2)

    # Copy PAPER_UPDATE_SUMMARY.md
    shutil.copy("outputs/aggregate/PAPER_UPDATE_SUMMARY.md", os.path.join(base_out, "PAPER_UPDATE_SUMMARY.md"))
    print("Paper assets generated successfully under outputs/paper_assets/")

if __name__ == "__main__":
    generate_paper_assets()
