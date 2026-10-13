"""What the model looks at: Grad-CAM on selected images and 2D maps of its embeddings."""

import math

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
from matplotlib.lines import Line2D
from pytorch_grad_cam import (EigenCAM, GradCAM, GradCAMPlusPlus, HiResCAM, LayerCAM,
                              ScoreCAM)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from sklearn.decomposition import PCA

import evaluation

CAM_METHODS = {
    "gradcam": GradCAM,
    "gradcam++": GradCAMPlusPlus,
    "hirescam": HiResCAM,
    "layercam": LayerCAM,
    "eigencam": EigenCAM,
    "scorecam": ScoreCAM,
}

# Colour-blind safe reference palette: series 1 = blue, series 2 = orange.
_BLUE, _ORANGE = "#2a78d6", "#eb6834"
_INK, _MUTED, _CONTEXT = "#52514e", "#898781", "#c3c2b7"


def compute_cam(model, images, target_layer, class_idx, method="gradcam"):
    """CAM maps (B, H, W) in [0, 1] for class `class_idx[i]` of each image.

    Each map is normalised on its own image: it shows where the model looks, not how
    confident it is.
    """
    model.eval()
    targets = [ClassifierOutputTarget(int(c)) for c in class_idx]
    with CAM_METHODS[method](model=model, target_layers=[target_layer]) as cam:
        return cam(input_tensor=images, targets=targets)


def cam_overlay(cam, color=_ORANGE, max_alpha=0.75):
    """Single-hue RGBA overlay whose opacity follows the map, so the image stays readable."""
    rgba = np.zeros((*cam.shape, 4), dtype=np.float32)
    rgba[..., :3] = mcolors.to_rgb(color)
    rgba[..., 3] = cam * max_alpha
    return rgba


def show_gradcam(model, dataset, indices, target_layer, classes, targets=("pred", "true"),
                 method="gradcam", true_labels=None, title=""):
    """Grid with the image, then one CAM row per target ("pred" or "true" class).

    `indices` are positions in a non-augmented `dataset` (= index of result["predictions"]).
    `true_labels` replaces the displayed true class (e.g. label_orig in the binary task).
    """
    indices = [int(i) for i in indices]
    if not indices:
        return None
    device = next(model.parameters()).device
    x = torch.stack([dataset[i][0] for i in indices])
    y = np.array([int(dataset[i][1]) for i in indices])
    with torch.no_grad():
        probs = torch.softmax(model.eval()(x.to(device)), dim=1).cpu().numpy()
    y_pred = probs.argmax(1)

    class_for = {"pred": y_pred, "true": y}
    row_names = {"pred": "CAM → predicted", "true": "CAM → true"}
    cams = {t: compute_cam(model, x, target_layer, class_for[t], method) for t in targets}
    shown = list(true_labels) if true_labels is not None else [classes[c] for c in y]

    n_rows, n_cols = 1 + len(targets), len(indices)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.3 * n_cols + 1.0, 2.3 * n_rows + 1.0),
                             squeeze=False)
    for j in range(n_cols):
        img = x[j, 0].numpy()
        axes[0, j].imshow(img, cmap="gray")
        axes[0, j].set_title(f"{shown[j]}\n→ {classes[y_pred[j]]}\n{probs[j, y_pred[j]]:.2f}",
                             fontsize=7, color=_INK)
        for r, t in enumerate(targets, start=1):
            axes[r, j].imshow(img, cmap="gray")
            axes[r, j].imshow(cam_overlay(cams[t][j]))

    for r, ax_row in enumerate(axes):
        for ax in ax_row:
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        ax_row[0].set_ylabel("image" if r == 0 else row_names[targets[r - 1]], fontsize=8, color=_MUTED)

    fig.suptitle(f"{title}\n{method}, orange = evidence for the class".strip(),
                 fontsize=10, color=_INK)
    fig.tight_layout()
    plt.close(fig)
    return fig


@torch.no_grad()
def extract_embeddings(model, loader, layer):
    """Flattened output of `layer` over a loader, with labels and model predictions."""
    device = next(model.parameters()).device
    model.eval()
    captured, labels, preds = [], [], []
    handle = layer.register_forward_hook(lambda m, inp, out: captured.append(out.flatten(1).cpu()))
    try:
        for x, y in loader:
            out = model(x.to(device))
            labels.append(y.numpy())
            preds.append(out.argmax(1).cpu().numpy())
    finally:
        handle.remove()
    return torch.cat(captured).numpy(), np.concatenate(labels), np.concatenate(preds)


def embedding_map(embeddings, method="umap", seed=0):
    """2D projection ("umap" | "tsne" | "pca"), after a 50-dim PCA to remove noise."""
    x = np.asarray(embeddings, dtype=np.float32)
    n_comp = min(50, x.shape[1], x.shape[0])
    x = PCA(n_components=2 if method == "pca" else n_comp, random_state=seed).fit_transform(x)
    if method == "pca":
        return x
    if method == "umap":
        import umap  # local import: numba takes seconds to load
        return umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=seed).fit_transform(x)
    if method == "tsne":
        from sklearn.manifold import TSNE
        return TSNE(n_components=2, perplexity=30, random_state=seed).fit_transform(x)
    raise ValueError(f"Unknown method: {method}")


def plot_embedding_map(xy, groups, correct, title=""):
    """Small multiples: one panel per group, its points coloured over the full grey cloud.

    Panels are sorted by size; errors are drawn in orange on top of correct predictions.
    """
    ncols = 4
    groups = np.asarray(groups).astype(str)
    names = pd.Series(groups).value_counts().index.tolist()
    nrows = math.ceil(len(names) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.9 * nrows + 0.6), squeeze=False)

    for ax, name in zip(axes.flat, names):
        m = groups == name
        ok, ko = m & correct, m & ~correct
        ax.scatter(xy[~m, 0], xy[~m, 1], s=2, color=_CONTEXT, alpha=0.35, linewidths=0, rasterized=True)
        ax.scatter(xy[ok, 0], xy[ok, 1], s=6, color=_BLUE, linewidths=0, rasterized=True)
        ax.scatter(xy[ko, 0], xy[ko, 1], s=10, color=_ORANGE, linewidths=0, rasterized=True)
        ax.set_title(f"{name}\nn = {m.sum()} · {correct[m].mean():.0%} correct",
                     fontsize=9, color=_INK, loc="left")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(_CONTEXT)
    for ax in axes.flat[len(names):]:
        ax.axis("off")

    handles = [Line2D([], [], ls="", marker="o", ms=6, color=_BLUE, label="panel class, correct"),
               Line2D([], [], ls="", marker="o", ms=6, color=_ORANGE, label="panel class, error"),
               Line2D([], [], ls="", marker="o", ms=5, color=_CONTEXT, label="other images")]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 1.0 - 0.25 / fig.get_figheight()))
    if title:
        fig.suptitle(title, fontsize=11, color=_INK, y=1.0 + 0.1 / fig.get_figheight())
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.6 / fig.get_figheight()))
    plt.close(fig)
    return fig


def log_figures(run_id, figures):
    """Attach figures to an existing run, e.g. {"explain/umap_val.png": fig}."""
    mlflow.set_tracking_uri(f"sqlite:///{evaluation.MLFLOW_DB}")
    client = mlflow.MlflowClient()
    for path, fig in figures.items():
        if fig is not None:
            client.log_figure(run_id, fig, path)
