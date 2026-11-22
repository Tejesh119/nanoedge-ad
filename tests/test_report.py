"""Tests for metrics.json schema validation and documentation placeholder checks."""
import os
import json
import pytest

def test_metrics_json_schema():
    """Verify metrics.json exists and strictly conforms to required schema keys."""
    metrics_path = "outputs/metrics.json"
    assert os.path.exists(metrics_path), f"metrics.json missing at {metrics_path}"

    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    expected_keys = [
        "metadata",
        "main_comparison_table",
        "quantization",
        "pruning_sweep",
        "runtime_simulation",
        "hardware_benchmarks",
        "per_anomaly_recall",
        "component_ablation",
        "acceptance_criteria",
        "research_claims_evaluation",
        "aggregate",
        "per_seed"
    ]
    for key in expected_keys:
        assert key in data, f"Key '{key}' missing from metrics.json schema"

    # Metadata checks
    assert data["metadata"]["project"] == "nanoedge-ad"
    assert len(data["metadata"]["seeds"]) == 5
    assert data["metadata"]["synthetic_data"] is True
    assert "aggregate" in data
    assert "per_seed" in data
    assert len(data["per_seed"]) == 5

    # Acceptance criteria checks
    assert len(data["acceptance_criteria"]) >= 7

def test_docs_contain_no_placeholders():
    """Verify KEY_METRICS.md and RUN_REQUIREMENTS.md contain zero unrendered placeholders."""
    forbidden_tokens = ["{{", "}}", "TODO", "TBD", "PLACEHOLDER", "FIXME", "XXX"]

    docs_to_check = [
        "docs/KEY_METRICS.md",
        "docs/RUN_REQUIREMENTS.md",
        "outputs/paper_draft.md"
    ]

    for doc in docs_to_check:
        assert os.path.exists(doc), f"Document {doc} must exist"
        with open(doc, "r", encoding="utf-8") as f:
            content = f.read()
        
        assert len(content.strip()) > 500, f"Document {doc} is too short"
        for token in forbidden_tokens:
            assert token not in content, f"Forbidden placeholder '{token}' found in {doc}"

def test_results_tables_exist():
    """Verify consolidated results tables exist and contain all models."""
    assert os.path.exists("outputs/results.csv")
    assert os.path.exists("outputs/results.md")

    with open("outputs/results.md", "r", encoding="utf-8") as f:
        md_text = f.read()

    assert "NanoEdge-AD TCN (INT8)" in md_text
    assert "B1_DecisionTree" in md_text
    assert "B2_VanillaCNN (FP32)" in md_text
    assert "B3_DenseAutoencoder" in md_text
    assert "B4_StaticPrunedCNN" in md_text

def test_table_numbers_match_aggregate():
    """Verify metrics in metrics.json and results.csv match outputs/aggregate/ exactly."""
    import pandas as pd
    agg_df = pd.read_csv("outputs/aggregate/table2_main_comparison.csv").set_index("model")
    results_df = pd.read_csv("outputs/results.csv").set_index("model")

    with open("outputs/metrics.json", "r", encoding="utf-8") as f:
        metrics = json.load(f)

    # Check that results.csv matches table2_main_comparison exactly
    for model in agg_df.index:
        assert model in results_df.index, f"Model {model} missing in results.csv"
        assert agg_df.loc[model, "f1"] == results_df.loc[model, "f1"], (
            f"F1 mismatch for {model}: {agg_df.loc[model, 'f1']} vs {results_df.loc[model, 'f1']}"
        )

    # Check metrics.json main comparison table matches aggregate
    for row in metrics["main_comparison_table"]:
        m = row["model"]
        if m in agg_df.index:
            assert row["f1_mean"] == round(float(agg_df.loc[m, "f1_mean"]), 4)
            assert row["f1_std"] == round(float(agg_df.loc[m, "f1_std"]), 4)


