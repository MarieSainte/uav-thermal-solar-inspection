"""Évaluation commune à tous les modèles, suivie dans MLflow.

Un seul code de métriques pour tout le projet : scripts/train.py l'appelle après
l'entraînement, et un modèle custom (ex. détection d'anomalie type PatchCore) l'appelle
directement avec ses propres sorties. Les métriques et artefacts portent les mêmes noms
dans les deux cas, donc les runs restent comparables dans MLflow.

Deux formats de sortie acceptés :
  - `probs` en 2D (n, n_classes) : probabilités, la prédiction est l'argmax ;
  - `probs` en 1D (n,) : score de la classe positive classes[1] (binaire uniquement),
    prédiction = score >= threshold. Le score n'a pas besoin d'être dans [0, 1].

Usage depuis un notebook de notebook/ :
    import sys; sys.path.insert(0, "../scripts")
    from evaluation import find_threshold, run_evaluation

    # Seuil choisi sur la validation, jamais sur le test
    threshold = find_threshold(val_labels, val_scores)

    result = run_evaluation(
        test_labels, test_scores,
        classes=["No-Anomaly", "Anomaly"],
        experiment_name="infrared-solar-custom-cnn-binary",
        run_name="patchcore-wrn50",
        threshold=threshold,
        dataframe=test,
        params={"backbone": "wide_resnet50_2"},
    )
"""

from __future__ import annotations

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


# ----------------------------------------------------------------
# CONFIGURATION MLFLOW
# ----------------------------------------------------------------
def setup_mlflow(experiment_name: str) -> str:
    """Pointe MLflow sur le SQLite du projet et active l'expérience (créée si besoin).

    Chemins résolus depuis ce fichier et non depuis le cwd : le script marche pareil
    lancé depuis notebook/ ou depuis la racine.
    """
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")
    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(experiment_name, artifact_location=f"file:///{MLFLOW_ARTIFACTS}")
    mlflow.set_experiment(experiment_name)
    return mlflow.get_tracking_uri()


def log_metrics_if_active(metrics, step=None):
    """Logue uniquement s'il y a un run ouvert, pour pouvoir appeler les fonctions hors MLflow."""
    if mlflow.active_run() is not None:
        mlflow.log_metrics(metrics, step=step)


def log_splits():
    """Joint les CSV de split au run, pour rejouer exactement le même découpage plus tard."""
    for nom in ("train", "val", "test"):
        path = DATA_DIR / f"{nom}.csv"
        if path.exists():
            mlflow.log_artifact(str(path), artifact_path="splits")


# ----------------------------------------------------------------
# SEUIL (sorties en score)
# ----------------------------------------------------------------
def find_threshold(labels, scores):
    """Seuil qui maximise le F1 de la classe positive. À calculer sur la validation."""
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    # La dernière paire precision/recall n'a pas de seuil associé.
    return float(thresholds[np.argmax(f1[:-1])])


# ----------------------------------------------------------------
# MÉTRIQUES
# ----------------------------------------------------------------
def evaluate_predictions(labels, probs, classes, split="test", dataframe=None, threshold=0.5):
    """Métriques, matrice de confusion, AUC ROC et table des prédictions d'un split.

    `dataframe` doit être dans le même ordre que `labels` : il sert à rattacher chaque
    prédiction à son image (colonnes id, filepath, label_orig si présentes).
    Si un run MLflow est ouvert, métriques et artefacts y sont loggés.
    """
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs, dtype=float)
    n_classes = len(classes)
    scores_mode = probs.ndim == 1
    if scores_mode and n_classes != 2:
        raise ValueError("Un score 1D n'a de sens qu'en binaire (classes[1] = classe positive).")

    # Score de la classe positive en binaire (1D fourni directement, ou colonne 1 des probas).
    pos_scores = probs if scores_mode else (probs[:, 1] if n_classes == 2 else None)
    preds = (pos_scores >= threshold).astype(int) if scores_mode else probs.argmax(1)

    metrics = {
        f"{split}_accuracy": accuracy_score(labels, preds),
        f"{split}_f1_macro": f1_score(labels, preds, average="macro"),
        f"{split}_f1_weighted": f1_score(labels, preds, average="weighted"),
    }

    # ===== AUC ROC =====
    # Multi-classes : global en one-vs-rest macro, plus une AUC par classe (les rares se
    # noient dans la moyenne). Binaire : AUC classique sur le score de la classe positive,
    # qui est classes[1] — à ordonner en conséquence dans le notebook.
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
        print(f"ROC AUC : non calculable ({e})")

    if n_classes == 2:
        pos = classes[1]
        metrics[f"{split}_precision_{pos}"] = precision_score(labels, preds, zero_division=0)
        metrics[f"{split}_recall_{pos}"] = recall_score(labels, preds, zero_division=0)
    if scores_mode:
        metrics[f"{split}_threshold"] = float(threshold)

    for i, cls in enumerate(classes):
        y_bin = (labels == i).astype(int)
        # Une classe absente du split n'a pas d'AUC définie.
        if not 0 < y_bin.sum() < len(y_bin):
            continue
        if n_classes == 2:
            # En binaire, l'AUC de la classe négative est la même que celle de la positive.
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

    # ===== Matrice de confusion =====
    # Brute (effectifs) renvoyée pour l'analyse, normalisée par ligne pour la figure.
    cm = confusion_matrix(labels, preds, labels=list(range(n_classes)))
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    cm_df.index.name, cm_df.columns.name = "true", "pred"

    cm_norm = confusion_matrix(labels, preds, labels=list(range(n_classes)), normalize="true")
    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay(cm_norm, display_labels=classes).plot(
        ax=ax, cmap="Blues", xticks_rotation=90, values_format=".2f", colorbar=False
    )
    ax.set_title(f"Matrice de confusion — {split} (normalisée par classe réelle)")
    fig.tight_layout()

    # ===== Prédictions =====
    # Une ligne par image : de quoi filtrer les bons / mauvais et regarder les images.
    predictions = pd.DataFrame({
        "y_true": labels,
        "y_pred": preds,
        "label": [classes[i] for i in labels],
        "pred_label": [classes[i] for i in preds],
    })
    if scores_mode:
        # Score brut : pas une proba, donc pas de "confidence" ni de prob_* par classe.
        predictions["score"] = pos_scores
        predictions["correct"] = predictions["y_true"] == predictions["y_pred"]
    else:
        predictions["confidence"] = probs.max(1)
        predictions["prob_true"] = probs[np.arange(len(labels)), labels]
        predictions["correct"] = predictions["y_true"] == predictions["y_pred"]
        probs_df = pd.DataFrame(probs, columns=[f"prob_{c}" for c in classes])
        predictions = pd.concat([predictions, probs_df], axis=1)

    if dataframe is not None and len(dataframe) == len(predictions):
        # label_orig : classe d'origine quand les labels ont été regroupés (cas binaire).
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

    # Fermée ici pour ne pas s'afficher en double ; le notebook l'affiche via result["figure"].
    plt.close(fig)

    return {
        "metrics": metrics,
        "confusion_matrix": cm_df,
        "roc_auc": roc_auc,
        "predictions": predictions,
        "report": txt_report,
        "figure": fig,
    }


# ----------------------------------------------------------------
# LANCEMENT D'UN RUN D'ÉVALUATION SEUL
# ----------------------------------------------------------------
def run_evaluation(labels, probs, classes, experiment_name, run_name=None, split="test",
                   dataframe=None, threshold=0.5, params=None, tags=None, log_split_files=True,
                   extra_metrics=None):
    """Un run MLflow d'évaluation pour un modèle entraîné ailleurs (ex. PatchCore custom).

    `params` / `tags` : tout ce qui décrit la technique. `extra_metrics` : métriques propres
    au modèle (ex. temps d'inférence, taille de la memory bank) loggées telles quelles.
    Renvoie le dict de `evaluate_predictions` enrichi de `run_id`.
    """
    setup_mlflow(experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({
            "n_classes": len(classes),
            f"n_{split}": len(labels),
            "output": "score" if np.asarray(probs).ndim == 1 else "probs",
            **(params or {}),
        })
        mlflow.set_tags(tags or {})
        if log_split_files:
            log_splits()
        if extra_metrics:
            mlflow.log_metrics(extra_metrics)

        print(f"===== {split.upper()} =====")
        result = evaluate_predictions(labels, probs, classes, split=split,
                                      dataframe=dataframe, threshold=threshold)

        print(f"\nRun MLflow : {run.info.run_id}")
        print(f"Visualiser : mlflow ui --backend-store-uri \"{mlflow.get_tracking_uri()}\"")

    result["run_id"] = run.info.run_id
    return result


def compare_runs(experiment_name, order_by="metrics.test_f1_macro DESC"):
    """Historique des runs d'une expérience, trié, pour comparer les techniques."""
    setup_mlflow(experiment_name)
    runs = mlflow.search_runs(experiment_names=[experiment_name], order_by=[order_by])
    colonnes = ["run_id", "tags.mlflow.runName", "metrics.best_val_f1_macro",
                "metrics.test_f1_macro", "metrics.test_roc_auc_ovr", "metrics.test_roc_auc",
                "metrics.test_accuracy", "params.model", "params.lr", "params.num_epochs",
                "start_time"]
    return runs[[c for c in colonnes if c in runs.columns]]
