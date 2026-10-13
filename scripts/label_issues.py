"""Label-error detection with cleanlab, from out-of-fold predicted probabilities.

A model's probabilities on its own training set are overfitted and confirm the wrong labels
it memorised, so every image must be predicted by a model that has not seen it.
"""

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

_INK = "#52514e"
META_COLS = ("id", "filepath", "split", "label_orig", "similarity_group")


def cross_val_pred_probs(df, make_model, make_optimizer, criterion, make_loader, n_splits=5,
                         num_epochs=20, stratify_col="label", device=None, seed=0):
    """Out-of-fold probabilities (N, n_classes) for every row of `df`, and the fold of each row.

    make_model() returns a fresh model, make_optimizer(model) its optimizer, and
    make_loader(df, train) a DataLoader (train=False: no augmentation, order kept).
    Folds keep similarity groups together, otherwise a near-duplicate left in training leaks
    its (possibly wrong) label. The epoch count is fixed: picking the best epoch with the
    suspected labels would bias the analysis.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = df.reset_index(drop=True)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    probs, folds = None, np.full(len(df), -1)

    for k, (tr_idx, ho_idx) in enumerate(sgkf.split(df, y=df[stratify_col], groups=df["similarity_group"])):
        print(f"\n===== Fold {k + 1}/{n_splits}: train {len(tr_idx)} | predict {len(ho_idx)} =====")
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


def find_label_issues_df(labels, pred_probs, classes, df=None):
    """One row per image, most suspicious first (index = position in `df`).

    label_quality is cleanlab's self-confidence score (low = suspicious). is_label_issue uses
    prune_by_noise_rate: the number of issues is estimated from the confident joint, not from
    an arbitrary threshold.
    """
    labels = np.asarray(labels).astype(int)
    pred_probs = np.asarray(pred_probs, dtype=np.float32)
    # n_jobs=1: cleanlab multiprocessing is fragile on Windows / Jupyter.
    is_issue = find_label_issues(labels, pred_probs, filter_by="prune_by_noise_rate", n_jobs=1)
    out = pd.DataFrame({
        "given_label": [classes[i] for i in labels],
        "suggested_label": [classes[i] for i in pred_probs.argmax(1)],
        "prob_given": pred_probs[np.arange(len(labels)), labels],
        "label_quality": get_label_quality_scores(labels, pred_probs, method="self_confidence"),
        "is_label_issue": is_issue,
    })
    if df is not None:
        df = df.reset_index(drop=True)
        out = pd.concat([df[[c for c in META_COLS if c in df]], out], axis=1)
    return out.sort_values("label_quality")


def label_noise_summary(labels, pred_probs, classes):
    """Overall health score, estimated joint (given label × true class) and per-class quality."""
    labels = np.asarray(labels).astype(int)
    health = overall_label_health_score(labels, pred_probs, verbose=False)
    joint = pd.DataFrame(estimate_joint(labels, pred_probs),
                         index=[f"given: {c}" for c in classes], columns=[f"true: {c}" for c in classes])
    per_class = rank_classes_by_label_quality(labels, pred_probs, class_names=classes)
    return health, joint, per_class


def show_issues(issues, n=30, title=""):
    """Grid of suspicious images: given label (and original type), model suggestion, quality."""
    ncols = 10
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
        lines += [f"→ {r['suggested_label']}", f"quality {r['label_quality']:.3f}"]
        ax.set_title("\n".join(lines), fontsize=7, color=_INK)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    for ax in axes.flat[len(rows):]:
        ax.axis("off")
    fig.suptitle(f"{title}\ngiven label → model suggestion".strip(), fontsize=10, color=_INK)
    fig.tight_layout()
    plt.close(fig)
    return fig


def log_label_issues(issues, labels, pred_probs, classes, experiment_name, run_name=None,
                     params=None, dataframe=None, figures=None):
    """MLflow run of the analysis: out-of-fold metrics (oof_ prefix), issue counts, tables."""
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
        print(f"\nMLflow run: {run.info.run_id}")
    return run.info.run_id


def _path_key(p):
    return os.path.normcase(os.path.abspath(str(p)))


def tag_fiftyone(issues, dataset_name="InfraredSolar", tag="label_issue"):
    """Tag the suspects in FiftyOne and add `label_quality` / `cleanlab_suggested` fields.

    Samples are matched by file path because FiftyOne ids change when the dataset is rebuilt.
    Returns the suspects view, lowest quality first.
    """
    import fiftyone as fo  # local import: slow to load, only needed for the review

    ds = fo.load_dataset(dataset_name)
    if not ds.persistent:
        print(f"Warning: '{dataset_name}' is not persistent, FiftyOne will delete it (and the tags) "
              "on the next start without another session. Set dataset.persistent = True to keep it.")

    id_by_path = {_path_key(p): i for p, i in zip(*ds.values(["filepath", "id"]))}
    sample_ids = issues["filepath"].map(lambda p: id_by_path.get(_path_key(p)))
    found = sample_ids.notna()
    if (~found).any():
        print(f"{(~found).sum()} images not found in '{dataset_name}' (skipped).")
    issues, sample_ids = issues[found], sample_ids[found]

    ds.untag_samples(tag)
    ds.set_values("label_quality", dict(zip(sample_ids, issues["label_quality"].astype(float))),
                  key_field="id")
    ds.set_values("cleanlab_suggested", dict(zip(sample_ids, issues["suggested_label"])), key_field="id")
    ds.select(sample_ids[issues["is_label_issue"]].tolist()).tag_samples(tag)
    return ds.match_tags(tag).sort_by("label_quality")
