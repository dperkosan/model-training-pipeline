from __future__ import annotations

import argparse
from pathlib import Path

from model_training_pipeline.artifacts import (
    load_json,
    load_model,
    load_split_indices,
    save_json,
)
from model_training_pipeline.data import load_dataset
from model_training_pipeline.metrics import evaluate_binary_classifier
from model_training_pipeline.train import predict_probabilities


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a saved run directory.")
    p.add_argument(
        "--run-dir", type=str, required=True, help="Path to a run folder under runs/"
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override threshold (default: from config.json or 0.5)",
    )
    p.add_argument(
        "--save", action="store_true", help="Save eval_metrics.json into the run folder"
    )
    p.add_argument("--debug", action="store_true", help="Print predict_proba details")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)

    model_path = run_dir / "model.joblib"
    config_path = run_dir / "config.json"
    splits_path = run_dir / "splits.json"
    eval_metrics_path = run_dir / "eval_metrics.json"

    # 1) Load model + splits
    model = load_model(path=model_path)
    _, _, idx_test = load_split_indices(path=splits_path)

    # 2) Load dataset and rebuild test set exactly
    X, y = load_dataset()
    X_test = X[idx_test]
    y_test = y[idx_test]

    # 3) Pick threshold: CLI override > config.json > default 0.5
    threshold: float
    if args.threshold is not None:
        threshold = float(args.threshold)
    else:
        cfg = load_json(path=config_path)
        threshold = 0.5
        if isinstance(cfg, dict):
            t = cfg.get("threshold")
            if isinstance(t, (int, float)):
                threshold = float(t)

    # 4) Predict scores + compute metrics
    test_scores = predict_probabilities(model=model, X=X_test, debug=args.debug)
    metrics = evaluate_binary_classifier(
        y_true=y_test, y_score=test_scores, threshold=threshold
    )

    print("Run dir:", str(run_dir))
    print("Test metrics:", metrics)

    # 5) Optional: save metrics as an artifact too
    if args.save:
        save_json(path=eval_metrics_path, obj={"test": metrics})
        print("Saved:", str(eval_metrics_path))


if __name__ == "__main__":
    main()
