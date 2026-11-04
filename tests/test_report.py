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
        "research_claims_evaluation"
    ]
    for key in expected_keys:
        assert key in data, f"Key '{key}' missing from metrics.json schema"

    # Metadata checks
    assert data["metadata"]["project"] == "nanoedge-ad"
    assert len(data["metadata"]["seeds"]) == 3
    assert data["metadata"]["synthetic_data"] is True

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

