"""Détection d'erreurs de label avec cleanlab, à partir de probabilités hors échantillon.

cleanlab compare le label donné à ce que prédit un modèle qui n'a PAS vu l'image : une
image dont le label reçoit une proba très faible, alors que le modèle est sûr d'une autre
classe, est suspecte. Les probas doivent donc être hors échantillon — celles d'un modèle
sur son propre train sont sur-apprises et cachent justement les labels faux qu'il a mémorisés.

Deux sources de probas :
  - les prédictions test d'un run (déjà hors échantillon) : immédiat, mais ne couvre que le test ;
  - `cross_val_pred_probs` : K modèles, chacun prédit le pli qu'il n'a pas vu -> une proba
    hors échantillon pour CHAQUE image. Les plis respectent les groupes de similarité :
    sinon un quasi-doublon resté dans le train "souffle" son label (même faux) au pli prédit.

Usage depuis un notebook de notebook/ :
    from label_issues import cross_val_pred_probs, find_label_issues_df

    oof_probs, folds = cross_val_pred_probs(all_df, make_model, make_optimizer, criterion, make_loader)
    issues = find_label_issues_df(all_df["label"].map(label2idx), oof_probs, classes, df=all_df)
    issues[issues["is_label_issue"]]
"""

from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
from cleanlab.count import estimate_joint
from cleanlab.dataset import overall_label_health_score, rank_classes_by_label_quality
from cleanlab.filter import find_label_issues
from cleanlab.rank import get_label_quality_scores
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

from evaluation import evaluate_predictions, setup_mlflow
from train import predict, train

_INK, _MUTED = "#52514e", "#898781"
META_COLS = ("id", "filepath", "split", "label_orig", "similarity_group")


# ----------------------------------------------------------------
# PROBAS HORS ÉCHANTILLON (validation croisée)
# ----------------------------------------------------------------
def cross_val_pred_probs(df, make_model, make_optimizer, criterion, make_loader, n_splits=5,
                         num_epochs=20, group_col="similarity_group", stratify_col="label",
                         device=None, seed=0):
    """Probas out-of-fold (N, n_classes) pour chaque ligne de `df`, et le pli de chacune.

    make_model()             -> nn.Module neuf (non entraîné)
    make_optimizer(model)    -> optimiseur
    make_loader(df, train)   -> DataLoader ; train=True : augmentation + shuffle,
                                train=False : sans augmentation, shuffle=False (ordre conservé)

    Nombre d'epochs fixe, sans sélection sur le pli prédit : choisir la meilleure epoch
    avec les labels qu'on soupçonne biaiserait justement l'analyse.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = df.reset_index(drop=True)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    probs, folds = None, np.full(len(df), -1)

    for k, (tr_idx, ho_idx) in enumerate(sgkf.split(df, y=df[stratify_col], groups=df[group_col])):
        print(f"\n===== Pli {k + 1}/{n_splits} : entraînement {len(tr_idx)} | prédit {len(ho_idx)} =====")
        torch.manual_seed(seed + k)
        model = make_model().to(device)
        optimizer = make_optimizer(model)
        train(model, criterion, optimizer, make_loader(df.iloc[tr_idx], True), val_loader=None,
              num_epochs=num_epochs, device=device, log_grads_every=0)
        _, _, p, _ = predict(model, make_loader(df.iloc[ho_idx], False), device)
        if probs is None:
            probs = np.zeros((len(df), p.shape[1]), dtype=np.float32)
        probs[ho_idx], folds[ho_idx] = p, k

    return probs, folds


# ----------------------------------------------------------------
# DÉTECTION
# ----------------------------------------------------------------
def find_label_issues_df(labels, pred_probs, classes, df=None, method="self_confidence"):
    """Une ligne par image, triée de la plus suspecte à la moins suspecte (index = position dans df).

    - label_quality   : score cleanlab dans [0, 1], bas = suspect (self_confidence = proba du
                        label donné ; "normalized_margin" pénalise aussi une autre classe très sûre)
    - is_label_issue  : sélection cleanlab (prune_by_noise_rate). Le NOMBRE d'erreurs est estimé
                        par la matrice « label donné × vraie classe » (confident joint), pas par
                        un seuil arbitraire ; on retire ensuite les plus suspectes de chaque case.
    - suggested_label : classe la plus probable selon le modèle
    """
    labels = np.asarray(labels).astype(int)
    pred_probs = np.asarray(pred_probs, dtype=np.float32)
    # n_jobs=1 : le multiprocessing de cleanlab est fragile sous Windows / Jupyter.
    is_issue = find_label_issues(labels, pred_probs, filter_by="prune_by_noise_rate", n_jobs=1)
    out = pd.DataFrame({
        "given_label": [classes[i] for i in labels],
        "suggested_label": [classes[i] for i in pred_probs.argmax(1)],
        "prob_given": pred_probs[np.arange(len(labels)), labels],
        "label_quality": get_label_quality_scores(labels, pred_probs, method=method),
        "is_label_issue": is_issue,
    })
    if df is not None:
        df = df.reset_index(drop=True)
        out = pd.concat([df[[c for c in META_COLS if c in df]], out], axis=1)
    return out.sort_values("label_quality")


def label_noise_summary(labels, pred_probs, classes):
    """Score de santé global, matrice jointe estimée et bruit par classe.

    joint[i, j] = part estimée du dataset labellisée i alors que la vraie classe est j.
    Hors diagonale = erreurs de label estimées, dans chaque sens.
    """
    labels = np.asarray(labels).astype(int)
    health = overall_label_health_score(labels, pred_probs, verbose=False)
    joint = pd.DataFrame(estimate_joint(labels, pred_probs),
                         index=[f"donné : {c}" for c in classes], columns=[f"vrai : {c}" for c in classes])
    per_class = rank_classes_by_label_quality(labels, pred_probs, class_names=classes)
    return health, joint, per_class


# ----------------------------------------------------------------
# VISUALISATION
# ----------------------------------------------------------------
def show_issues(issues, n=30, ncols=10, title=""):
    """Grille des images suspectes : label donné (et type d'origine), avis du modèle, qualité."""
    rows = issues.head(n)
    if rows.empty:
        return None
    nrows = math.ceil(len(rows) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.3 * ncols + 0.4, 2.7 * nrows + 0.6), squeeze=False)
    for ax, (_, r) in zip(axes.flat, rows.iterrows()):
        ax.imshow(Image.open(r["filepath"]).convert("L"), cmap="gray")
        lines = [r["given_label"]]
        if "label_orig" in r and r["label_orig"] != r["given_label"]:
            lines.append(f"({r['label_orig']})")
        lines += [f"→ {r['suggested_label']}", f"qualité {r['label_quality']:.3f}"]
        ax.set_title("\n".join(lines), fontsize=7, color=_INK)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    for ax in axes.flat[len(rows):]:
        ax.axis("off")
    fig.suptitle(f"{title}\nlabel donné → classe suggérée par le modèle".strip(), fontsize=10, color=_INK)
    fig.tight_layout()
    plt.close(fig)
    return fig


# ----------------------------------------------------------------
# MLFLOW / FIFTYONE
# ----------------------------------------------------------------
def log_label_issues(issues, labels, pred_probs, classes, experiment_name, run_name=None,
                     params=None, dataframe=None, figures=None):
    """Run MLflow de l'analyse : métriques out-of-fold (mêmes noms que les autres runs, préfixe
    oof_), nombre d'erreurs suspectées, table complète et probas en artefacts."""
    setup_mlflow(experiment_name)
    health, joint, per_class = label_noise_summary(labels, pred_probs, classes)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({"n_samples": len(labels), **(params or {})})
        evaluate_predictions(labels, pred_probs, classes, split="oof", dataframe=dataframe)
        mlflow.log_metrics({
            "n_label_issues": int(issues["is_label_issue"].sum()),
            "label_issue_rate": float(issues["is_label_issue"].mean()),
            "label_health_score": float(health),
        })
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            issues.to_csv(tmp / "label_issues.csv", index=False)
            joint.to_csv(tmp / "estimated_joint.csv")
            per_class.to_csv(tmp / "label_quality_per_class.csv", index=False)
            np.save(tmp / "oof_pred_probs.npy", pred_probs)
            for f in tmp.iterdir():
                mlflow.log_artifact(str(f), artifact_path="cleanlab")
        for path, fig in (figures or {}).items():
            if fig is not None:
                mlflow.log_figure(fig, path)
        print(f"\nRun MLflow : {run.info.run_id}")
    return run.info.run_id


def _path_key(p):
    """Chemin comparable d'une machine / d'un cwd à l'autre (casse du lecteur, séparateurs)."""
    return os.path.normcase(os.path.abspath(str(p)))


def tag_fiftyone(issues, dataset_name="InfraredSolar", tag="label_issue"):
    """Marque les suspects dans FiftyOne pour la revue visuelle, et ajoute à toutes les images
    analysées les champs `label_quality` et `cleanlab_suggested`.

    Correspondance par chemin de fichier et non par `id` : les ids FiftyOne changent dès que
    le dataset est reconstruit (relance de l'EDA), les chemins non.
    Le tag est d'abord retiré partout, pour qu'un nouveau passage remplace l'ancien.
    Renvoie la vue des suspects triée par qualité croissante (à ouvrir avec fo.launch_app).
    """
    import fiftyone as fo  # import local : lent à charger, inutile sans revue

    ds = fo.load_dataset(dataset_name)
    if not ds.persistent:
        print(f"Attention : '{dataset_name}' n'est pas persistant. FiftyOne le supprime au prochain "
              "démarrage sans autre session ouverte, tags compris (dataset.persistent = True pour le garder).")

    id_by_path = {_path_key(p): i for p, i in zip(*ds.values(["filepath", "id"]))}
    sample_ids = issues["filepath"].map(lambda p: id_by_path.get(_path_key(p)))
    found = sample_ids.notna()
    if (~found).any():
        print(f"{(~found).sum()} images introuvables dans '{dataset_name}' (ignorées).")
    issues, sample_ids = issues[found], sample_ids[found]

    ds.untag_samples(tag)
    ds.set_values("label_quality", dict(zip(sample_ids, issues["label_quality"].astype(float))),
                  key_field="id")
    ds.set_values("cleanlab_suggested", dict(zip(sample_ids, issues["suggested_label"])), key_field="id")
    ds.select(sample_ids[issues["is_label_issue"]].tolist()).tag_samples(tag)
    return ds.match_tags(tag).sort_by("label_quality")
