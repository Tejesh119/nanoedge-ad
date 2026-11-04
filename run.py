"""Cross-platform pipeline execution runner matching Makefile targets."""
import sys
import subprocess

TARGETS = {
    "setup": [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
    "data": [sys.executable, "-m", "src.data_gen"],
    "train": [sys.executable, "-m", "src.train"],
    "quantize": [sys.executable, "-m", "src.quantize"],
    "prune": [sys.executable, "-m", "src.prune_experiment"],
    "simulate": [sys.executable, "-m", "src.runtime_pruning"],
    "benchmark": [sys.executable, "-m", "src.benchmark"],
    "evaluate": [sys.executable, "-m", "src.evaluate"],
    "ablation": [sys.executable, "-m", "src.ablation"],
    "report": [sys.executable, "-m", "src.report"],
    "test": [sys.executable, "-m", "pytest", "tests/", "-v"],
}

ALL_SEQUENCE = ["data", "train", "quantize", "prune", "simulate", "benchmark", "evaluate", "ablation", "report", "test"]

def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "all"
    if target == "all":
        for t in ALL_SEQUENCE:
            print(f"\n==================== RUNNING TARGET: {t} ====================")
            cmd = TARGETS[t]
            ret = subprocess.call(cmd)
            if ret != 0:
                print(f"Error: Target '{t}' exited with code {ret}")
                sys.exit(ret)
        print("\nAll pipeline targets executed successfully!")
    elif target in TARGETS:
        print(f"Running target: {target}")
        ret = subprocess.call(TARGETS[target])
        sys.exit(ret)
    else:
        print(f"Unknown target: {target}. Available targets: {list(TARGETS.keys()) + ['all']}")
        sys.exit(1)

if __name__ == "__main__":
    main()

