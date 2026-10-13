"""Metrics, thresholding and MLflow logging shared by every model of the project.

`probs` can be 2D (n, n_classes) probabilities, predicted with argmax, or 1D scores of the
positive class classes[1] (binary only), predicted with `score >= threshold`.
"""

import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MLFLOW_DB = (PROJECT_ROOT / "mlflow.db").as_posix()
MLFLOW_ARTIFACTS = (PROJECT_ROOT / "mlartifacts").as_posix()


def setup_mlflow(experiment_name):
    """Point MLflow at the project SQLite store and activate the experiment (created if needed)."""
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")
    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(experiment_name, artifact_location=f"file:///{MLFLOW_ARTIFACTS}")
    mlflow.set_experiment(experiment_name)


def log_metrics_if_active(metrics, step=None):
    if mlflow.active_run() is not None:
        mlflow.log_metrics(metrics, step=step)


def log_splits():
    """Attach the split CSVs to the active run, so the exact split can be replayed."""
    for name in ("train", "val", "test"):
        path = DATA_DIR / f"{name}.csv"
        if path.exists():
            mlflow.log_artifact(str(path), artifact_path="splits")


def find_threshold(labels, scores):
    """Threshold maximising the positive-class F1. Compute it on validation, never on test."""
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    # The last precision/recall pair has no threshold.
    return float(thresholds[np.argmax(f1[:-1])])


def evaluate_predictions(labels, probs, classes, split="test", dataframe=None, threshold=0.5):
    """Metrics, confusion matrix, ROC AUC and per-image predictions for one split.

    `dataframe` must follow the order of `labels`; its id / filepath / label_orig columns are
    copied into the predictions. Metrics and artifacts are logged if an MLflow run is active.
    """
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs, dtype=float)
    n_classes = len(classes)
    scores_mode = probs.ndim == 1
    if scores_mode and n_classes != 2:
        raise ValueError("1D scores are only supported for binary tasks (classes[1] = positive).")

    pos_scores = probs if scores_mode else (probs[:, 1] if n_classes == 2 else None)
    preds = (pos_scores >= threshold).astype(int) if scores_mode else probs.argmax(1)

    metrics = {
        f"{split}_accuracy": accuracy_score(labels, preds),
        f"{split}_f1_macro": f1_score(labels, preds, average="macro"),
        f"{split}_f1_weighted": f1_score(labels, preds, average="weighted"),
    }

    roc_auc = {"global": None, "per_class": {}}
    try:
        if n_classes == 2:
            roc_auc["global"] = roc_auc_score(labels, pos_scores)
            metrics[f"{split}_roc_auc"] = roc_auc["global"]
        else:
            roc_auc["global"] = roc_auc_score(labels, probs, multi_class="ovr", average="macro",
                                              labels=list(range(n_classes)))
            metrics[f"{split}_roc_auc_ovr"] = roc_auc["global"]
    except ValueError as e:
        print(f"ROC AUC: not computable ({e})")

    if n_classes == 2:
        pos = classes[1]
        metrics[f"{split}_precision_{pos}"] = precision_score(labels, preds, zero_division=0)
        metrics[f"{split}_recall_{pos}"] = recall_score(labels, preds, zero_division=0)
    if scores_mode:
        metrics[f"{split}_threshold"] = float(threshold)

    for i, cls in enumerate(classes):
        y_bin = (labels == i).astype(int)
        if not 0 < y_bin.sum() < len(y_bin):
            continue
        if n_classes == 2:
            roc_auc["per_class"][cls] = roc_auc["global"]
        else:
            roc_auc["per_class"][cls] = roc_auc_score(y_bin, probs[:, i])

    for k, v in metrics.items():
        print(f"{k:22s} : {v:.4f}")

    report = classification_report(labels, preds, labels=list(range(n_classes)),
                                   target_names=classes, zero_division=0, output_dict=True)
    for cls in classes:
        metrics[f"{split}_f1_{cls}"] = report[cls]["f1-score"]
    for cls, auc in roc_auc["per_class"].items():
        if auc is not None:
            metrics[f"{split}_auc_{cls}"] = auc

    txt_report = classification_report(labels, preds, labels=list(range(n_classes)),
                                       target_names=classes, zero_division=0)
    print()
    print(txt_report)

    cm = confusion_matrix(labels, preds, labels=list(range(n_classes)))
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    cm_df.index.name, cm_df.columns.name = "true", "pred"

    cm_norm = confusion_matrix(labels, preds, labels=list(range(n_classes)), normalize="true")
    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay(cm_norm, display_labels=classes).plot(
        ax=ax, cmap="Blues", xticks_rotation=90, values_format=".2f", colorbar=False
    )
    ax.set_title(f"Confusion matrix, {split} (normalised by true class)")
    fig.tight_layout()

    predictions = pd.DataFrame({
        "y_true": labels,
        "y_pred": preds,
        "label": [classes[i] for i in labels],
        "pred_label": [classes[i] for i in preds],
    })
    if scores_mode:
        predictions["score"] = pos_scores
        predictions["correct"] = predictions["y_true"] == predictions["y_pred"]
    else:
        predictions["confidence"] = probs.max(1)
        predictions["prob_true"] = probs[np.arange(len(labels)), labels]
        predictions["correct"] = predictions["y_true"] == predictions["y_pred"]
        probs_df = pd.DataFrame(probs, columns=[f"prob_{c}" for c in classes])
        predictions = pd.concat([predictions, probs_df], axis=1)

    if dataframe is not None and len(dataframe) == len(predictions):
        meta_cols = [c for c in ("id", "filepath", "label_orig") if c in dataframe]
        meta = dataframe.reset_index(drop=True)[meta_cols]
        predictions = pd.concat([meta, predictions], axis=1)

    log_metrics_if_active(metrics)
    if mlflow.active_run() is not None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            fig.savefig(tmp / f"confusion_matrix_{split}.png", dpi=120)
            (tmp / f"classification_report_{split}.txt").write_text(txt_report, encoding="utf-8")
            cm_df.to_csv(tmp / f"confusion_matrix_{split}.csv")
            predictions.to_csv(tmp / f"predictions_{split}.csv", index=False)
            for f in tmp.iterdir():
                mlflow.log_artifact(str(f), artifact_path="evaluation")

    # Closed so it is not displayed twice; the notebook shows result["figure"].
    plt.close(fig)

    return {
        "metrics": metrics,
        "confusion_matrix": cm_df,
        "roc_auc": roc_auc,
        "predictions": predictions,
        "report": txt_report,
        "figure": fig,
    }


def run_evaluation(labels, probs, classes, experiment_name, run_name=None,
                   dataframe=None, threshold=0.5, params=None, tags=None, extra_metrics=None):
    """MLflow test-evaluation run for a model trained elsewhere (e.g. PatchCore).

    `extra_metrics` are model-specific values logged as is (inference time, memory bank size...).
    """
    setup_mlflow(experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({
            "n_classes": len(classes),
            "n_test": len(labels),
            "output": "score" if np.asarray(probs).ndim == 1 else "probs",
            **(params or {}),
        })
        mlflow.set_tags(tags or {})
        log_splits()
        if extra_metrics:
            mlflow.log_metrics(extra_metrics)

        print("===== TEST =====")
        result = evaluate_predictions(labels, probs, classes, split="test",
                                      dataframe=dataframe, threshold=threshold)

        print(f"\nMLflow run: {run.info.run_id}")
        print(f"View with: mlflow ui --backend-store-uri \"{mlflow.get_tracking_uri()}\"")

    result["run_id"] = run.info.run_id
    return result


def compare_runs(experiment_name):
    """Runs of an experiment, best test macro F1 first."""
    setup_mlflow(experiment_name)
    runs = mlflow.search_runs(experiment_names=[experiment_name],
                              order_by=["metrics.test_f1_macro DESC"])
    columns = ["run_id", "tags.mlflow.runName", "metrics.best_val_f1_macro",
               "metrics.test_f1_macro", "metrics.test_roc_auc_ovr", "metrics.test_roc_auc",
               "metrics.test_accuracy", "params.model", "params.lr", "params.num_epochs",
               "start_time"]
    return runs[[c for c in columns if c in runs.columns]]
