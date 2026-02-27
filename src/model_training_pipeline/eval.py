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
from model_training_pipeline.utils import get_tracer, setup_logging, setup_tracing


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
    p.add_argument(
        "--log-level", type=str, default="INFO", help="DEBUG, INFO, WARNING, ERROR"
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    logger = setup_logging(level=args.log_level, log_file=run_dir / "eval.log")
    setup_tracing(service_name="model-training-pipeline", logger=logger)
    tracer = get_tracer("model_training_pipeline.eval")
    logger.info("Evaluating run: %s", run_dir)

    model_path = run_dir / "model.joblib"
    config_path = run_dir / "config.json"
    splits_path = run_dir / "splits.json"
    eval_metrics_path = run_dir / "eval_metrics.json"

    with tracer.start_as_current_span("eval.run") as run_span:
        run_span.set_attribute("run.dir", str(run_dir))
        run_span.set_attribute("eval.log_level", args.log_level.upper())

        # 1) Load model + splits
        with tracer.start_as_current_span("artifacts.load"):
            model = load_model(path=model_path)
            _, _, idx_test = load_split_indices(path=splits_path)

        # 2) Load dataset and rebuild test set exactly
        with tracer.start_as_current_span("data.load"):
            X, y = load_dataset()
            X_test = X[idx_test]
            y_test = y[idx_test]

        # 3) Pick threshold: CLI override > config.json > default 0.5
        threshold: float
        if args.threshold is not None:
            threshold = float(args.threshold)
            run_span.set_attribute("eval.threshold_source", "cli")
        else:
            cfg = load_json(path=config_path)
            threshold = 0.5
            run_span.set_attribute("eval.threshold_source", "default")
            if isinstance(cfg, dict):
                t = cfg.get("threshold")
                if isinstance(t, (int, float)):
                    threshold = float(t)
                    run_span.set_attribute("eval.threshold_source", "config")

        run_span.set_attribute("eval.threshold", threshold)

        # 4) Predict scores + compute metrics
        with tracer.start_as_current_span("model.evaluate"):
            test_scores = predict_probabilities(model=model, X=X_test)
            metrics = evaluate_binary_classifier(
                y_true=y_test, y_score=test_scores, threshold=threshold
            )

    logger.info("Run dir: %s", run_dir)
    logger.info("Test metrics: %s", metrics)

    # 5) Optional: save metrics as an artifact too
    if args.save:
        save_json(path=eval_metrics_path, obj={"test": metrics})
        logger.info("Saved: %s", eval_metrics_path)


if __name__ == "__main__":
    main()
