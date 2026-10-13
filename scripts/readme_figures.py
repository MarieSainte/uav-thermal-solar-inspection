"""Regenerate the README figures from the logged MLflow runs (nothing is retrained).

Usage:
    uv run python scripts/readme_figures.py
    uv run python scripts/readme_figures.py --binary-run <run_id> --out docs/figures
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluation import DATA_DIR, MLFLOW_DB, PROJECT_ROOT   # noqa: E402
from explain import cam_overlay, compute_cam              # noqa: E402

# Runs analysed in the notebooks.
BINARY_RUN = "e3b730f6d7da4ca58cfe187e82ab38fb"       # simplecnn-binary-weighted-aug
MULTI_RUN = "879e9e710b164088b94c34bad8ebeebd"        # simplecnn-weighted-aug (12 classes)
PATCHCORE_RUN = "839ba7c402e7433eba615ddd4d9d2556"    # patchcore-autoencoder-pca64-core0.1

# Colour-blind safe reference palette: series 1 = blue, series 2 = orange.
SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"
# Sequential blue ramp, from the surface colour (≈ 0) to the darkest step.
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", [SURFACE, "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])


class SimpleCNN(nn.Module):
    """Same definition as the notebooks: the MLflow pickle references __main__.SimpleCNN."""

    def __init__(self, n_classes=12, in_ch=1):
        super().__init__()

        def block(i, o):
            return [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                    nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                    nn.MaxPool2d(2)]

        self.features = nn.Sequential(*block(in_ch, 32), *block(32, 64), *block(64, 128))
        self.pool = nn.AdaptiveAvgPool2d((5, 3))
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(128 * 5 * 3, n_classes))

    def forward(self, x):
        return self.head(self.pool(self.features(x)).flatten(1))


# ----------------------------------------------------------------
# STYLE
# ----------------------------------------------------------------
def set_style():
    plt.rcParams.update({
        "font.family": ["Segoe UI", "DejaVu Sans"],
        "font.size": 10,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "text.color": INK, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.edgecolor": BASELINE, "axes.linewidth": 1,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "axes.grid": False, "grid.color": GRID, "grid.linewidth": 1,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 11, "axes.titleweight": "semibold", "axes.titlelocation": "left",
        "legend.frameon": False,
    })


def save(fig, out_dir, name):
    path = out_dir / name
    fig.savefig(path, dpi=160, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"  {path.name}")


def wrap(cls):
    """Break long class names at the last hyphen."""
    head, _, tail = cls.rpartition("-")
    return f"{head}-\n{tail}" if len(cls) > 10 and head else cls


def load_image(path):
    return np.asarray(Image.open(path).convert("L"))


def predictions(run_id):
    path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="evaluation/predictions_test.csv")
    return pd.read_csv(path)


# ----------------------------------------------------------------
# FIGURES
# ----------------------------------------------------------------
def class_gallery(multi, out_dir, per_class=3):
    """Most confident correct examples of each class (12-class model)."""
    classes = multi["label"].value_counts().index.tolist()
    fig, axes = plt.subplots(per_class, len(classes), figsize=(1.05 * len(classes), 1.75 * per_class + 0.4))
    for j, cls in enumerate(classes):
        best = multi[(multi["label"] == cls) & multi["correct"]].nlargest(per_class, "prob_true")
        for i, (_, r) in enumerate(best.iterrows()):
            ax = axes[i, j]
            ax.imshow(load_image(r["filepath"]), cmap="inferno", vmin=0, vmax=255)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
        axes[0, j].set_title(wrap(cls), fontsize=8.5, loc="center", fontweight="normal", color=INK_2)
    fig.suptitle("12 classes, 40×24 px thermal crops (inferno colormap)", x=0.01, ha="left",
                 fontsize=11, fontweight="semibold")
    fig.tight_layout()
    save(fig, out_dir, "class_gallery.png")


def binary_results(cnn, patchcore, out_dir):
    """CNN confusion matrix and ROC curves of the CNN vs PatchCore on the same test set."""
    fig, (ax_cm, ax_roc) = plt.subplots(1, 2, figsize=(10, 4.3), gridspec_kw={"width_ratios": [1, 1.15]})

    classes = ["No-Anomaly", "Anomaly"]
    cm = confusion_matrix(cnn["y_true"], cnn["y_pred"])
    cm_norm = cm / cm.sum(1, keepdims=True)
    ax_cm.imshow(cm_norm, cmap=SEQ, vmin=0, vmax=1)
    for (i, j), v in np.ndenumerate(cm):
        ax_cm.text(j, i, f"{cm_norm[i, j]:.1%}\n{v:,} images", ha="center", va="center", fontsize=10,
                   color="white" if cm_norm[i, j] > 0.5 else INK)
    ax_cm.set_xticks([0, 1], classes); ax_cm.set_yticks([0, 1], classes)
    ax_cm.set_xlabel("Predicted"); ax_cm.set_ylabel("True")
    ax_cm.tick_params(length=0)
    for s in ax_cm.spines.values():
        s.set_visible(False)
    ax_cm.set_title("Custom CNN: confusion matrix (test)")

    for df, col, name, color in [(cnn, "prob_Anomaly", "Custom CNN (supervised)", BLUE),
                                 (patchcore, "score", "PatchCore (unsupervised)", ORANGE)]:
        fpr, tpr, _ = roc_curve(df["y_true"], df[col])
        auc = roc_auc_score(df["y_true"], df[col])
        ax_roc.plot(fpr, tpr, color=color, lw=2, solid_capstyle="round", label=f"{name}, AUC {auc:.3f}")
    ax_roc.plot([0, 1], [0, 1], color=BASELINE, lw=1)
    ax_roc.set_xlim(0, 1); ax_roc.set_ylim(0, 1.01)
    ax_roc.set_xlabel("False positive rate"); ax_roc.set_ylabel("True positive rate (recall)")
    ax_roc.grid(True); ax_roc.set_axisbelow(True)
    ax_roc.legend(loc="lower right", labelcolor=INK_2)
    ax_roc.set_title("ROC curves: anomaly vs healthy (test, n = 3,994)")
    fig.tight_layout(w_pad=3)
    save(fig, out_dir, "binary_results.png")


def recall_by_type(cnn, patchcore, out_dir):
    """Recall per original anomaly type for both binary models."""
    def recall(df):
        a = df[df["label"] == "Anomaly"]
        return a.groupby("label_orig")["correct"].agg(["mean", "size"])

    r_cnn, r_pc = recall(cnn), recall(patchcore)
    order = r_cnn.sort_values("mean").index
    y = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hlines(y, r_cnn.loc[order, "mean"], r_pc.loc[order, "mean"], color=BASELINE, lw=2, zorder=1)
    for r, other_r, color, name in [(r_cnn, r_pc, BLUE, "Custom CNN"), (r_pc, r_cnn, ORANGE, "PatchCore")]:
        v = r.loc[order, "mean"].values
        # Label on the outer side of the gap so it never overlaps the other dot.
        other = other_r.loc[order, "mean"].values
        ax.scatter(v, y, s=70, color=color, edgecolor=SURFACE, linewidth=2, zorder=3, label=name)
        for yi, vi, oi in zip(y, v, other):
            right = vi >= oi
            ax.text(vi + (0.018 if right else -0.018), yi, f"{vi:.0%}", va="center",
                    ha="left" if right else "right", fontsize=8.5, color=INK_2)
    ax.set_yticks(y, [f"{t}  (n={int(r_cnn.loc[t, 'size'])})" for t in order])
    ax.set_xlim(0.2, 1.08)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(True, axis="x"); ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Recall (share of anomalies flagged)")
    ax.legend(loc="lower right", labelcolor=INK_2, handletextpad=0.2)
    ax.set_title("Recall by anomaly type (binary task, test set)")
    fig.tight_layout()
    save(fig, out_dir, "recall_by_type.png")


def multiclass_confusion(multi, out_dir):
    """Row-normalised 12×12 confusion matrix; only cells ≥ 5% are labelled."""
    classes = sorted(multi["label"].unique())
    cm = confusion_matrix(multi["label"], multi["pred_label"], labels=classes)
    cm_norm = cm / cm.sum(1, keepdims=True)

    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = ax.imshow(cm_norm, cmap=SEQ, vmin=0, vmax=1)
    for (i, j), v in np.ndenumerate(cm_norm):
        if v >= 0.05:
            ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=8,
                    color="white" if v > 0.5 else INK)
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, format=matplotlib.ticker.PercentFormatter(1.0))
    cb.outline.set_visible(False)
    cb.ax.tick_params(color=MUTED, labelcolor=INK_2, length=0)
    ax.set_title("12-class confusion matrix (test, row-normalised; cells ≥ 5% labelled)")
    fig.tight_layout()
    save(fig, out_dir, "multiclass_confusion.png")


def gradcam_examples(model, cnn, device, out_dir):
    """One correctly detected anomaly per type: thermal image, then Grad-CAM for "Anomaly"."""
    picks = (cnn[cnn["correct"] & (cnn["label"] == "Anomaly")]
             .sort_values("confidence", ascending=False).groupby("label_orig").head(1)
             .sort_values("label_orig"))
    imgs = np.stack([load_image(p) for p in picks["filepath"]]).astype(np.float32)
    x = torch.from_numpy((imgs / 255.0 - 0.5) / 0.5).unsqueeze(1).to(device)
    cams = compute_cam(model, x, model.features[19], class_idx=[1] * len(picks))

    fig, axes = plt.subplots(2, len(picks), figsize=(1.15 * len(picks), 4.4))
    for j, (img, cam, cls) in enumerate(zip(imgs, cams, picks["label_orig"])):
        axes[0, j].imshow(img, cmap="inferno", vmin=0, vmax=255)
        axes[1, j].imshow(img, cmap="gray", vmin=0, vmax=255)
        axes[1, j].imshow(cam_overlay(cam))
        axes[0, j].set_title(wrap(cls), fontsize=8, loc="center", fontweight="normal", color=INK_2)
        for ax in axes[:, j]:
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
    axes[0, 0].set_ylabel("thermal", fontsize=8.5, color=MUTED)
    axes[1, 0].set_ylabel("Grad-CAM", fontsize=8.5, color=MUTED)
    fig.suptitle("Where the binary CNN looks: one correctly detected anomaly per type "
                 "(orange = evidence for “Anomaly”)", x=0.01, ha="left", fontsize=11, fontweight="semibold")
    fig.tight_layout()
    save(fig, out_dir, "gradcam_binary.png")


def label_issues(out_dir, n=10):
    """Most suspicious cleanlab candidates, in both directions."""
    path = DATA_DIR / "label_issues_binary.csv"
    if not path.exists():
        print("  (label_issues_binary.csv not found: run the cleanlab part of custom_cnn_binary)")
        return
    issues = pd.read_csv(path).sort_values("label_quality")
    rows = [("Labelled No-Anomaly, model is confident it is an anomaly",
             issues[issues["given_label"] == "No-Anomaly"].head(n)),
            ("Labelled as an anomaly, model sees a healthy module",
             issues[issues["given_label"] == "Anomaly"].head(n))]

    fig = plt.figure(figsize=(1.1 * n, 5.4), layout="constrained")
    for sub, (title, df) in zip(fig.subfigures(2, 1, hspace=0.06), rows):
        axes = sub.subplots(1, n)
        for ax, (_, r) in zip(axes, df.iterrows()):
            ax.imshow(load_image(r["filepath"]), cmap="inferno", vmin=0, vmax=255)
            ax.set_title(wrap(r["label_orig"]), fontsize=7.5, loc="center", fontweight="normal", color=INK_2)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
        sub.suptitle(title, x=0.01, ha="left", fontsize=9.5, color=INK_2)
    fig.suptitle("cleanlab label-issue candidates (out-of-fold, 5-fold group CV)",
                 x=0.01, ha="left", fontsize=11, fontweight="semibold")
    save(fig, out_dir, "label_issues.png")


def intensity_robustness(model, device, out_dir):
    """Binary CNN F1 when the test images are shifted in brightness or contrast.

    In non-radiometric thermal images the grey level depends on the camera settings,
    so a robust model should not move.
    """
    test = pd.read_csv(DATA_DIR / "test.csv")
    X = np.stack([load_image(p) for p in test["filepath"]]).astype(np.float32)
    y = (test["label"] != "No-Anomaly").astype(int).values
    mean = X.mean((1, 2), keepdims=True)

    @torch.no_grad()
    def f1(arr):
        t = torch.from_numpy((np.clip(arr, 0, 255) / 255.0 - 0.5) / 0.5).unsqueeze(1).float()
        p = torch.cat([model(t[i:i + 512].to(device)).softmax(1)[:, 1].cpu()
                       for i in range(0, len(t), 512)]).numpy()
        return f1_score(y, p > 0.5, average="macro")

    shifts = np.arange(-40, 41, 10)
    factors = np.round(np.arange(0.6, 1.41, 0.1), 2)
    f1_shift = [f1(X + d) for d in shifts]
    f1_contrast = [f1(mean + c * (X - mean)) for c in factors]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, xs, ys, ref, xlabel, title in [
        (axes[0], shifts, f1_shift, 0, "Brightness shift (grey levels, 0–255 scale)", "Brightness"),
        (axes[1], factors, f1_contrast, 1.0, "Contrast factor around the image mean", "Contrast"),
    ]:
        ax.plot(xs, ys, color=BLUE, lw=2, marker="o", ms=7, mec=SURFACE, mew=2, solid_capstyle="round")
        k = int(np.argmin(np.abs(np.asarray(xs) - ref)))
        ax.annotate(f"original\n{ys[k]:.3f}", (xs[k], ys[k]), xytext=(0, 12), textcoords="offset points",
                    ha="center", fontsize=8.5, color=INK_2)
        for k in (0, len(xs) - 1):
            ax.annotate(f"{ys[k]:.3f}", (xs[k], ys[k]), xytext=(0, -16), textcoords="offset points",
                        ha="center", fontsize=8.5, color=INK_2)
        ax.grid(True, axis="y"); ax.set_axisbelow(True)
        ax.set_xlabel(xlabel); ax.set_title(title)
    axes[0].set_ylabel("Test F1 (macro)")
    lo = min(min(f1_shift), min(f1_contrast))
    axes[0].set_ylim(lo - 0.03, max(max(f1_shift), max(f1_contrast)) + 0.025)
    fig.suptitle("Sensitivity of the binary CNN to thermal intensity changes (no intensity augmentation)",
                 x=0.01, ha="left", fontsize=11, fontweight="semibold")
    fig.tight_layout()
    save(fig, out_dir, "intensity_robustness.png")
    return dict(zip(shifts, f1_shift)), dict(zip(factors, f1_contrast))


# ----------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--binary-run", default=BINARY_RUN)
    parser.add_argument("--multi-run", default=MULTI_RUN)
    parser.add_argument("--patchcore-run", default=PATCHCORE_RUN)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "docs" / "figures")
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_style()

    cnn, multi, patchcore = (predictions(r) for r in (args.binary_run, args.multi_run, args.patchcore_run))
    model = mlflow.pytorch.load_model(f"runs:/{args.binary_run}/model", map_location=device).eval()

    print(f"Figures -> {args.out}")
    class_gallery(multi, args.out)
    binary_results(cnn, patchcore, args.out)
    recall_by_type(cnn, patchcore, args.out)
    multiclass_confusion(multi, args.out)
    gradcam_examples(model, cnn, device, args.out)
    label_issues(args.out)
    shift, contrast = intensity_robustness(model, device, args.out)
    print("F1 vs brightness:", {int(k): round(v, 4) for k, v in shift.items()})
    print("F1 vs contrast:  ", {float(k): round(v, 4) for k, v in contrast.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
