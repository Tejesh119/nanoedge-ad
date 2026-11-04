"""INT8 Post-Training Quantization and Validation for NanoEdge-AD."""
import os
import json
import argparse
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

from src.utils import load_config, setup_logger, set_seed
from src.windowing import prepare_dataset
from src.models import SensorGate

logger = setup_logger("quantize")

def run_tflite_inference(tflite_model_bytes: bytes, X: np.ndarray) -> tuple:
    """Run inference with a quantized TFLite model using the TFLite Interpreter.
    
    Handles input and output quantization scaling/zero-point automatically.
    """
    interpreter = tf.lite.Interpreter(model_content=tflite_model_bytes)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    in_scale, in_zero_point = input_details["quantization"]
    out_scale, out_zero_point = output_details["quantization"]

    probs = []
    for i in range(len(X)):
        sample = X[i:i+1].astype(np.float32)
        if input_details["dtype"] == np.int8:
            if in_scale > 0:
                quant_sample = np.clip(np.round(sample / in_scale) + in_zero_point, -128, 127).astype(np.int8)
            else:
                quant_sample = sample.astype(np.int8)
            interpreter.set_tensor(input_details["index"], quant_sample)
        else:
            interpreter.set_tensor(input_details["index"], sample)

        interpreter.invoke()
        out = interpreter.get_tensor(output_details["index"])

        if output_details["dtype"] == np.int8 and out_scale > 0:
            dequant_out = (out.astype(np.float32) - out_zero_point) * out_scale
        else:
            dequant_out = out.astype(np.float32)

        # dequant_out already contains calibrated class probabilities [p_norm, p_anom]
        prob_anom = float(dequant_out[0, 1])
        probs.append(prob_anom)

    quant_params = {
        "input_scale": float(in_scale),
        "input_zero_point": int(in_zero_point),
        "input_dtype": str(input_details["dtype"]),
        "output_scale": float(out_scale),
        "output_zero_point": int(out_zero_point),
        "output_dtype": str(output_details["dtype"])
    }
    return np.array(probs, dtype=np.float32), quant_params

def quantize_model(config_path: str = "config.yaml"):
    """Perform post-training INT8 quantization and verify accuracy against FP32 model."""
    cfg = load_config(config_path)
    primary_seed = cfg.get("primary_seed", 42)
    set_seed(primary_seed)

    ds = prepare_dataset(cfg, seed=primary_seed)
    fp32_path = cfg["paths"]["model_fp32"]
    int8_path = cfg["paths"]["model_int8"]
    quant_info_path = cfg["paths"]["quant_info"]

    logger.info(f"Loading FP32 model from {fp32_path}...")
    model = tf.keras.models.load_model(fp32_path, custom_objects={"SensorGate": SensorGate})

    # Evaluate FP32 model on test split
    fp32_test_probs = model.predict(ds["X_test"], verbose=0)[:, 1]
    # Optimal threshold from validation set
    fp32_val_probs = model.predict(ds["X_val"], verbose=0)[:, 1]
    thresholds = np.linspace(0.05, 0.95, 91)
    best_thresh = 0.5
    best_f1 = -1.0
    for t in thresholds:
        sc = f1_score(ds["y_val"], (fp32_val_probs >= t).astype(int), zero_division=0)
        if sc > best_f1:
            best_f1 = sc
            best_thresh = float(t)

    fp32_preds = (fp32_test_probs >= best_thresh).astype(int)
    fp32_f1 = float(f1_score(ds["y_test"], fp32_preds, zero_division=0))
    fp32_acc = float(accuracy_score(ds["y_test"], fp32_preds))
    logger.info(f"FP32 Baseline Test F1: {fp32_f1:.4f} (Threshold: {best_thresh:.3f})")

    # Calibration loop
    calib_sizes = [600, 1000, 1500]
    tflite_bytes = None
    int8_probs = None
    quant_params = {}
    best_agreement = 0.0

    for calib_n in calib_sizes:
        logger.info(f"Attempting INT8 quantization with {calib_n} calibration windows...")
        def rep_dataset():
            for i in range(min(calib_n, len(ds["X_train"]))):
                yield [ds["X_train"][i:i+1].astype(np.float32)]

        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = rep_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.TFLITE_BUILTINS]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

        candidate_bytes = converter.convert()
        cand_probs, cand_params = run_tflite_inference(candidate_bytes, ds["X_test"])
        cand_preds = (cand_probs >= best_thresh).astype(int)
        cand_agreement = float(np.mean(cand_preds == fp32_preds) * 100.0)
        cand_f1 = float(f1_score(ds["y_test"], cand_preds, zero_division=0))
        f1_drop = fp32_f1 - cand_f1

        logger.info(f"Calib {calib_n}: INT8 F1 = {cand_f1:.4f} (Drop: {f1_drop:+.4f}), Agreement = {cand_agreement:.2f}%")

        tflite_bytes = candidate_bytes
        int8_probs = cand_probs
        quant_params = cand_params
        best_agreement = cand_agreement

        if f1_drop <= 0.03 and cand_agreement >= 95.0:
            logger.info("Quantization acceptance criteria met!")
            break
        else:
            logger.warning(f"Drop {f1_drop:.4f} > 0.03 or agreement {cand_agreement:.1f}% < 95%. Retrying with larger calibration...")

    # Save INT8 TFLite model
    os.makedirs(os.path.dirname(int8_path), exist_ok=True)
    with open(int8_path, "wb") as f:
        f.write(tflite_bytes)

    model_size_bytes = len(tflite_bytes)
    model_size_kb = model_size_bytes / 1024.0

    int8_preds = (int8_probs >= best_thresh).astype(int)
    int8_f1 = float(f1_score(ds["y_test"], int8_preds, zero_division=0))
    int8_acc = float(accuracy_score(ds["y_test"], int8_preds))
    int8_prec = float(precision_score(ds["y_test"], int8_preds, zero_division=0))
    int8_rec = float(recall_score(ds["y_test"], int8_preds, zero_division=0))
    
    cm = confusion_matrix(ds["y_test"], int8_preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    int8_fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    int8_auc = float(roc_auc_score(ds["y_test"], int8_probs))

    agreement_pct = float(np.mean(int8_preds == fp32_preds) * 100.0)
    f1_delta = float(abs(fp32_f1 - int8_f1))

    criteria = {
        "param_count_under_15k": bool(model.count_params() < 15000),
        "int8_file_under_60kb": bool(model_size_kb < 60.0),
        "f1_within_3_pts": bool(f1_delta <= 0.03),
        "agreement_gte_95_pct": bool(agreement_pct >= 95.0)
    }

    quant_info = {
        "model_name": "TCN_SensorGate_INT8",
        "fp32_params": int(model.count_params()),
        "model_size_bytes": int(model_size_bytes),
        "model_size_kb": float(round(model_size_kb, 2)),
        "fp32_f1": float(round(fp32_f1, 4)),
        "int8_f1": float(round(int8_f1, 4)),
        "f1_delta": float(round(f1_delta, 4)),
        "fp32_accuracy": float(round(fp32_acc, 4)),
        "int8_accuracy": float(round(int8_acc, 4)),
        "int8_precision": float(round(int8_prec, 4)),
        "int8_recall": float(round(int8_rec, 4)),
        "int8_fpr": float(round(int8_fpr, 4)),
        "int8_roc_auc": float(round(int8_auc, 4)),
        "prediction_agreement_pct": float(round(agreement_pct, 2)),
        "quantization_parameters": quant_params,
        "acceptance_criteria": criteria
    }

    with open(quant_info_path, "w") as f:
        json.dump(quant_info, f, indent=2)

    logger.info(f"Saved INT8 model to: {int8_path} ({model_size_kb:.2f} KB)")
    logger.info(f"Saved quantization info to: {quant_info_path}")

    print("\n==================== INT8 QUANTIZATION SUMMARY ====================")
    print(f"FP32 Model Parameters:   {model.count_params():,}")
    print(f"INT8 Model File Size:    {model_size_kb:.2f} KB (Criterion < 60 KB: {'PASS' if criteria['int8_file_under_60kb'] else 'FAIL'})")
    print(f"FP32 F1:                 {fp32_f1:.4f}")
    print(f"INT8 F1:                 {int8_f1:.4f} (Delta: {f1_delta:.4f}, Criterion <= 0.03: {'PASS' if criteria['f1_within_3_pts'] else 'FAIL'})")
    print(f"Prediction Agreement:    {agreement_pct:.2f}% (Criterion >= 95%: {'PASS' if criteria['agreement_gte_95_pct'] else 'FAIL'})")
    print("===================================================================\n")

    return quant_info

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    quantize_model(args.config)

if __name__ == "__main__":
    main()
