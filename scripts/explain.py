"""Ce que le modèle voit : Grad-CAM sur des images choisies, carte UMAP des embeddings.

Fonctionne avec n'importe quel classifieur PyTorch : on lui indique la couche à expliquer
(Grad-CAM) et la couche dont la sortie sert d'embedding.

Usage depuis un notebook de notebook/ (après run_experiment) :
    from explain import show_gradcam, extract_embeddings, embedding_map, plot_embedding_map

    # Où le modèle regarde, pour la classe prédite et pour la vraie classe
    erreurs = result["predictions"].query("~correct").nlargest(10, "confidence")
    show_gradcam(model, dataset_test, erreurs.index, model.features[19], classes)

    # Embeddings du val set projetés en 2D, une classe par panneau
    emb, y, y_pred = extract_embeddings(model, val_loader, model.pool)
    xy = embedding_map(emb)
    plot_embedding_map(xy, groups=[classes[i] for i in y], correct=y == y_pred)
"""

from __future__ import annotations

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
    "hirescam": HiResCAM,      # plus fidèle que Grad-CAM : pas de moyenne des gradients par canal
    "layercam": LayerCAM,      # bon sur les couches peu profondes (cartes plus fines)
    "eigencam": EigenCAM,      # sans gradient ni classe : composante principale des activations
    "scorecam": ScoreCAM,      # sans gradient, par masquage : lent mais robuste
}

# Palette de référence (validée daltonisme) : série 1 = bleu, série 2 = orange.
_BLUE, _ORANGE = "#2a78d6", "#eb6834"
_INK, _MUTED, _CONTEXT = "#52514e", "#898781", "#c3c2b7"


# ----------------------------------------------------------------
# GRAD-CAM
# ----------------------------------------------------------------
def compute_cam(model, images, target_layer, class_idx, method="gradcam"):
    """Cartes CAM (B, H, W) dans [0, 1] pour la classe `class_idx[i]` de chaque image.

    Chaque carte est normalisée sur sa propre image : elle dit OÙ le modèle regarde,
    pas COMBIEN la classe est probable. Une carte vive sur une image à 5 % reste une carte vive.
    """
    model.eval()
    targets = [ClassifierOutputTarget(int(c)) for c in class_idx]
    with CAM_METHODS[method](model=model, target_layers=[target_layer]) as cam:
        return cam(input_tensor=images, targets=targets)


def _overlay(cam, color=_ORANGE, max_alpha=0.75):
    """Une seule teinte dont l'opacité suit la carte : le reste de l'image reste lisible."""
    rgba = np.zeros((*cam.shape, 4), dtype=np.float32)
    rgba[..., :3] = mcolors.to_rgb(color)
    rgba[..., 3] = cam * max_alpha
    return rgba


def show_gradcam(model, dataset, indices, target_layer, classes, targets=("pred", "true"),
                 method="gradcam", true_labels=None, title=""):
    """Grille : l'image, puis une ligne de CAM par cible ("pred" = classe prédite, "true" = vraie).

    Sur une erreur, comparer les deux lignes montre ce qui a attiré le modèle vers la
    mauvaise classe, et s'il a seulement regardé l'endroit de la vraie anomalie.

    `indices` : positions dans `dataset` (= index de result["predictions"]). Utiliser un
    dataset sans augmentation. `true_labels` : texte à afficher à la place de la vraie
    classe (ex. label_orig en binaire), aligné sur `indices`.
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
    row_names = {"pred": "CAM → prédite", "true": "CAM → vraie"}
    cams = {t: compute_cam(model, x, target_layer, class_for[t], method) for t in targets}
    shown = list(true_labels) if true_labels is not None else [classes[c] for c in y]

    n_rows, n_cols = 1 + len(targets), len(indices)
    # Vignettes 40×24 (portrait) : ~2.3 de haut par ligne pour 1.3 de large, + place des titres.
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.3 * n_cols + 1.0, 2.3 * n_rows + 1.0),
                             squeeze=False)
    for j in range(n_cols):
        img = x[j, 0].numpy()
        axes[0, j].imshow(img, cmap="gray")
        axes[0, j].set_title(f"{shown[j]}\n→ {classes[y_pred[j]]}\n{probs[j, y_pred[j]]:.2f}",
                             fontsize=7, color=_INK)
        for r, t in enumerate(targets, start=1):
            axes[r, j].imshow(img, cmap="gray")
            axes[r, j].imshow(_overlay(cams[t][j]))

    for r, ax_row in enumerate(axes):
        for ax in ax_row:
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        ax_row[0].set_ylabel("image" if r == 0 else row_names[targets[r - 1]], fontsize=8, color=_MUTED)

    fig.suptitle(f"{title}\n{method} — orange = zone qui pousse vers la classe".strip(),
                 fontsize=10, color=_INK)
    fig.tight_layout()
    plt.close(fig)
    return fig


# ----------------------------------------------------------------
# EMBEDDINGS
# ----------------------------------------------------------------
@torch.no_grad()
def extract_embeddings(model, loader, layer):
    """Sortie aplatie de `layer` sur tout un loader, avec labels et prédictions du modèle.

    Pour SimpleCNN, `model.pool` (5×3×128 = 1920 dims) est exactement ce que reçoit la tête
    de classification : l'espace dans lequel le modèle prend sa décision.
    `y_pred` vaut None si la sortie du modèle n'est pas des logits (B, n_classes).
    """
    device = next(model.parameters()).device
    model.eval()
    captured, labels, preds = [], [], []
    handle = layer.register_forward_hook(lambda m, inp, out: captured.append(out.flatten(1).cpu()))
    try:
        for x, y in loader:
            out = model(x.to(device))
            labels.append(y.numpy())
            if out.ndim == 2:
                preds.append(out.argmax(1).cpu().numpy())
    finally:
        handle.remove()
    y_pred = np.concatenate(preds) if preds else None
    return torch.cat(captured).numpy(), np.concatenate(labels), y_pred


def embedding_map(embeddings, method="umap", pca_dims=50, n_neighbors=15, min_dist=0.1, seed=0):
    """Projection 2D des embeddings ("umap" | "tsne" | "pca").

    Une PCA à `pca_dims` passe d'abord : elle retire le bruit des 1920 dims et accélère
    UMAP / t-SNE sans changer la structure globale.
    """
    x = np.asarray(embeddings, dtype=np.float32)
    n_comp = min(pca_dims, x.shape[1], x.shape[0])
    x = PCA(n_components=2 if method == "pca" else n_comp, random_state=seed).fit_transform(x)
    if method == "pca":
        return x
    if method == "umap":
        import umap  # import local : numba met plusieurs secondes à se charger
        return umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist,
                         random_state=seed).fit_transform(x)
    if method == "tsne":
        from sklearn.manifold import TSNE
        return TSNE(n_components=2, perplexity=30, random_state=seed).fit_transform(x)
    raise ValueError(f"method inconnue : {method}")


def plot_embedding_map(xy, groups, correct=None, ncols=4, title=""):
    """Petits multiples : un panneau par groupe, ses points en couleur sur le nuage complet en gris.

    Un seul nuage à 12 couleurs est illisible (et indistinguable pour un daltonien) ;
    un panneau par classe montre où elle vit, et si ses erreurs (orange) tombent dans
    le territoire d'une autre classe. Panneaux triés par effectif décroissant.
    """
    groups = np.asarray(groups).astype(str)
    names = pd.Series(groups).value_counts().index.tolist()
    nrows = math.ceil(len(names) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.9 * nrows + 0.6), squeeze=False)

    for ax, name in zip(axes.flat, names):
        m = groups == name
        ax.scatter(xy[~m, 0], xy[~m, 1], s=2, color=_CONTEXT, alpha=0.35, linewidths=0, rasterized=True)
        if correct is None:
            ax.scatter(xy[m, 0], xy[m, 1], s=6, color=_BLUE, linewidths=0, rasterized=True)
            sub = f"n = {m.sum()}"
        else:
            ok, ko = m & correct, m & ~correct
            ax.scatter(xy[ok, 0], xy[ok, 1], s=6, color=_BLUE, linewidths=0, rasterized=True)
            # Erreurs dessinées en dernier pour rester visibles au-dessus des bonnes.
            ax.scatter(xy[ko, 0], xy[ko, 1], s=10, color=_ORANGE, linewidths=0, rasterized=True)
            sub = f"n = {m.sum()} · {correct[m].mean():.0%} bien classées"
        ax.set_title(f"{name}\n{sub}", fontsize=9, color=_INK, loc="left")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(_CONTEXT)
    for ax in axes.flat[len(names):]:
        ax.axis("off")

    handles = [Line2D([], [], ls="", marker="o", ms=6, color=_BLUE,
                      label="classe du panneau" + (" — bien classée" if correct is not None else "")),
               Line2D([], [], ls="", marker="o", ms=5, color=_CONTEXT, label="autres images")]
    if correct is not None:
        handles.insert(1, Line2D([], [], ls="", marker="o", ms=6, color=_ORANGE,
                                 label="classe du panneau — erreur"))
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 1.0 - 0.25 / fig.get_figheight()))
    if title:
        fig.suptitle(title, fontsize=11, color=_INK, y=1.0 + 0.1 / fig.get_figheight())
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.6 / fig.get_figheight()))
    plt.close(fig)
    return fig


# ----------------------------------------------------------------
# MLFLOW
# ----------------------------------------------------------------
def log_figures(run_id, figures):
    """Attache des figures à un run existant, ex. {"explain/umap_val.png": fig}."""
    mlflow.set_tracking_uri(f"sqlite:///{evaluation.MLFLOW_DB}")
    client = mlflow.MlflowClient()
    for path, fig in figures.items():
        if fig is not None:
            client.log_figure(run_id, fig, path)
