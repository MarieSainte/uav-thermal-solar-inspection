"""Boucle d'entraînement + évaluation suivies dans MLflow, réutilisable d'un essai à l'autre.

Le notebook prépare les données, le modèle, la loss et l'optimiseur ; ce module fait le reste :
entraînement avec suivi de la validation, rechargement du meilleur modèle, évaluation
sur le test, logging MLflow, et renvoie de quoi analyser les prédictions.

Usage depuis un notebook de notebook/ :
    import sys; sys.path.insert(0, "../scripts")
    from train import run_experiment

    result = run_experiment(
        model, criterion, optimizer,
        train_loader, val_loader, test_loader,
        classes=classes,
        experiment_name="infrared-solar-custom-cnn",
        run_name="simplecnn-weighted-aug",
        num_epochs=15,
        params={"augmentation": "hflip+vflip"},
    )
    result["confusion_matrix"], result["roc_auc"], result["predictions"]
"""

from __future__ import annotations

import copy

import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import numpy as np
import torch
from mlflow.models import infer_signature
from sklearn.metrics import f1_score

# Métriques et config MLflow partagées avec les modèles évalués hors de ce script.
from evaluation import (compare_runs, evaluate_predictions, log_metrics_if_active,  # noqa: F401
                        log_splits, setup_mlflow)


# ----------------------------------------------------------------
# INFÉRENCE
# ----------------------------------------------------------------
def predict(model, loader, device, criterion=None):
    """Passe un loader dans le modèle. Renvoie labels, préds, probas et la loss moyenne."""
    model.eval()
    total_loss, all_labels, all_probs = 0.0, [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            if criterion is not None:
                total_loss += criterion(outputs, labels).item() * images.size(0)
            all_probs.append(torch.softmax(outputs, dim=1).cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    labels = np.concatenate(all_labels)
    probs = np.concatenate(all_probs)
    loss = total_loss / len(loader.dataset) if criterion is not None else None
    return labels, probs.argmax(1), probs, loss


# ----------------------------------------------------------------
# DYNAMIQUE D'ENTRAÎNEMENT (gradients)
# ----------------------------------------------------------------
@torch.no_grad()
def grad_norms(model):
    """Norme L2 du gradient de chaque paramètre + norme globale, en une seule synchro GPU.

    À appeler après backward() et AVANT un éventuel clipping : c'est la norme brute qui
    révèle une explosion. `grad/total` est la valeur que clip_grad_norm_ renverrait.
    """
    named = [(n, p.grad.norm()) for n, p in model.named_parameters() if p.grad is not None]
    if not named:
        return {}
    names, norms = zip(*named)
    norms = torch.stack(norms)
    values = torch.cat([norms, norms.norm()[None]]).tolist()
    metrics = {f"grad/{n}": v for n, v in zip(names, values)}
    metrics["grad/total"] = values[-1]
    return metrics


@torch.no_grad()
def update_ratios(model, before):
    """||Δw|| / ||w|| par paramètre, entre avant et après optimizer.step().

    Ordre de grandeur sain : ~1e-3. Bien en dessous, la couche n'apprend plus (lr trop
    faible, gradients qui s'éteignent) ; bien au-dessus, les poids sont bousculés à
    chaque pas (lr trop fort).
    """
    names, ratios = [], []
    for n, p in model.named_parameters():
        if n in before:
            names.append(n)
            ratios.append((p - before[n]).norm() / (before[n].norm() + 1e-8))
    return {f"update_ratio/{n}": v for n, v in zip(names, torch.stack(ratios).tolist())}


# Rampe ordinale bleue : clair = couche d'entrée, foncé = couche de sortie.
_DEPTH_RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
               "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
_INK, _MUTED, _GRID = "#52514e", "#898781", "#e1e0d9"


def plot_training_dynamics(run_id, model=None):
    """Normes de gradient et ratios de mise à jour d'un run, couche par couche.

    Avec `model`, seuls les noyaux (conv / linear, ndim > 1) sont tracés : biais et
    paramètres BatchNorm noieraient le graphe. Sans, tous les `.weight` logués.
    Les courbes complètes, biais compris, restent dans MLflow (préfixes grad/ et update_ratio/).
    """
    client = mlflow.MlflowClient()
    run = client.get_run(run_id)
    keys = set(run.data.metrics)
    steps_per_epoch = float(run.data.params.get("steps_per_epoch", 1))

    if model is not None:
        layers = [n for n, p in model.named_parameters() if p.ndim > 1]
    else:
        layers = [k.split("/", 1)[1] for k in sorted(keys)
                  if k.startswith("grad/") and k.endswith(".weight")]
    layers = [n for n in layers if f"grad/{n}" in keys]
    if not layers:
        raise ValueError("Aucune métrique grad/* dans ce run (log_grads_every=0 ?)")

    def history(key):
        h = sorted(client.get_metric_history(run_id, key), key=lambda m: m.step)
        return np.array([m.step for m in h]) / steps_per_epoch, np.array([m.value for m in h])

    ramp_idx = np.linspace(0, len(_DEPTH_RAMP) - 1, len(layers)).round().astype(int)
    colors = [_DEPTH_RAMP[i] for i in ramp_idx]
    single = _DEPTH_RAMP[4]

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    (ax_g, ax_u), (ax_t, ax_l) = axes
    for name, c in zip(layers, colors):
        ax_g.plot(*history(f"grad/{name}"), color=c, lw=2, label=name)
        ax_u.plot(*history(f"update_ratio/{name}"), color=c, lw=2, label=name)
    ax_u.axhline(1e-3, color=_MUTED, ls="--", lw=1)
    ax_u.annotate("1e-3", xy=(1, 1e-3), xycoords=("axes fraction", "data"),
                  xytext=(4, 0), textcoords="offset points", va="center", color=_MUTED, fontsize=8)

    ax_t.plot(*history("grad/total"), color=single, lw=2)
    clip = run.data.params.get("grad_clip", "none")
    if clip != "none":
        ax_t.axhline(float(clip), color=_MUTED, ls="--", lw=1)
        ax_t.annotate(f"clip = {clip}", xy=(1, float(clip)), xycoords=("axes fraction", "data"),
                      xytext=(4, 0), textcoords="offset points", va="center", color=_MUTED, fontsize=8)
    if "train_loss_batch" in keys:
        ax_l.plot(*history("train_loss_batch"), color=single, lw=2)

    panels = [(ax_g, "Norme du gradient par couche", True),
              (ax_u, "Ratio de mise à jour ||Δw|| / ||w||", True),
              (ax_t, "Norme globale du gradient (avant clipping)", True),
              (ax_l, "Loss du batch", False)]
    for ax, title, log in panels:
        ax.set_title(title, loc="left", color=_INK, fontsize=10)
        if log:
            ax.set_yscale("log")
        ax.grid(True, color=_GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(_MUTED)
        ax.tick_params(colors=_MUTED, labelsize=8)
    for ax in axes[1]:
        ax.set_xlabel("epoch", color=_MUTED, fontsize=9)

    handles, labels = ax_g.get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False,
               fontsize=8, title="couche (entrée → sortie)", title_fontsize=8)
    fig.tight_layout()
    # Fermée ici pour ne pas s'afficher en double ; le notebook l'affiche via le résultat.
    plt.close(fig)
    return fig


# ----------------------------------------------------------------
# BOUCLE D'ENTRAÎNEMENT
# ----------------------------------------------------------------
def train(model, criterion, optimizer, train_loader, val_loader=None, num_epochs=30,
          device="cpu", scheduler=None, log_grads_every=50, grad_clip=None):
    """Entraîne et recharge le meilleur état au sens du F1 macro de validation.

    `log_grads_every` : tous les combien de steps loguer normes de gradient, ratios de mise
    à jour et loss du batch (0 / None pour couper). `grad_clip` : norme max des gradients,
    None = pas de clipping.
    """
    best_f1, best_epoch, best_state = -1.0, -1, None
    step = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()

            # Pas à chaque step : ~50 métriques par écriture MLflow, et un clone des poids.
            monitor = bool(log_grads_every) and step % log_grads_every == 0
            if monitor:
                step_metrics = {"train_loss_batch": loss.item(), **grad_norms(model)}
                before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

            optimizer.step()

            if monitor:
                step_metrics.update(update_ratios(model, before))
                log_metrics_if_active(step_metrics, step=step)
            step += 1

            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        epoch_metrics = {"train_loss": train_loss, "lr": optimizer.param_groups[0]["lr"]}
        msg = f"Epoch [{epoch+1}/{num_epochs}] train_loss: {train_loss:.4f}"

        if val_loader is not None:
            y, y_pred, _, val_loss = predict(model, val_loader, device, criterion)
            val_f1 = f1_score(y, y_pred, average="macro")
            epoch_metrics.update({"val_loss": val_loss, "val_f1_macro": val_f1})
            msg += f" | val_loss: {val_loss:.4f} | val_f1_macro: {val_f1:.4f}"
            if val_f1 > best_f1:
                best_f1, best_epoch, best_state = val_f1, epoch, copy.deepcopy(model.state_dict())
                msg += "  <- best"

        # Le scheduler optionnel permet de tester des techniques de LR sans toucher à la boucle.
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(epoch_metrics.get("val_loss", train_loss))
            else:
                scheduler.step()

        log_metrics_if_active(epoch_metrics, step=epoch)
        print(msg)

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\nMeilleur modèle rechargé (epoch {best_epoch+1}, val_f1_macro = {best_f1:.4f})")
        log_metrics_if_active({"best_val_f1_macro": best_f1, "best_epoch": best_epoch})

    return best_f1


# ----------------------------------------------------------------
# ÉVALUATION
# ----------------------------------------------------------------
def evaluate(model, loader, classes, device, split="test", dataframe=None):
    """Évalue un modèle PyTorch sur un loader, via le code commun de scripts/evaluation.py.

    `dataframe` doit être dans le même ordre que le loader (shuffle=False). Par défaut
    on reprend `loader.dataset.dataframe` pour rattacher chaque prédiction à son image.
    """
    labels, _, probs, _ = predict(model, loader, device)
    if dataframe is None:
        dataframe = getattr(loader.dataset, "dataframe", None)
    return evaluate_predictions(labels, probs, classes, split=split, dataframe=dataframe)


# ----------------------------------------------------------------
# LANCEMENT D'UN RUN COMPLET
# ----------------------------------------------------------------
def run_experiment(model, criterion, optimizer, train_loader, val_loader, test_loader, classes,
                   experiment_name, run_name=None, num_epochs=15, device=None, scheduler=None,
                   params=None, tags=None, log_model=True, log_split_files=True,
                   log_grads_every=50, grad_clip=None):
    """Un run MLflow = entraînement + évaluation test du meilleur modèle.

    `params` et `tags` sont ajoutés à ceux déduits automatiquement : y mettre tout ce
    qui distingue la technique testée (augmentation, pondération, architecture...).
    `log_grads_every` / `grad_clip` : voir `train`.

    Renvoie le dict de `evaluate` enrichi de `run_id`, `best_val_f1` et, si les gradients
    sont logués, `dynamics_figure`.
    """
    if device is None:
        device = next(model.parameters()).device
    model.to(device)

    setup_mlflow(experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        auto_params = {
            "model": type(model).__name__,
            "n_params": sum(p.numel() for p in model.parameters()),
            "n_classes": len(classes),
            "optimizer": type(optimizer).__name__,
            "lr": optimizer.param_groups[0]["lr"],
            "weight_decay": optimizer.param_groups[0].get("weight_decay", 0),
            "criterion": type(criterion).__name__,
            "scheduler": type(scheduler).__name__ if scheduler is not None else "none",
            "batch_size_train": train_loader.batch_size,
            "batch_size_eval": test_loader.batch_size,
            "num_epochs": num_epochs,
            "steps_per_epoch": len(train_loader),
            "grad_clip": grad_clip if grad_clip is not None else "none",
            "log_grads_every": log_grads_every or 0,
            "n_train": len(train_loader.dataset),
            "n_val": len(val_loader.dataset) if val_loader is not None else 0,
            "n_test": len(test_loader.dataset),
            "device": str(device),
        }
        mlflow.log_params({**auto_params, **(params or {})})
        mlflow.set_tags(tags or {})

        if log_split_files:
            log_splits()

        best_f1 = train(model, criterion, optimizer, train_loader, val_loader,
                        num_epochs=num_epochs, device=device, scheduler=scheduler,
                        log_grads_every=log_grads_every, grad_clip=grad_clip)

        dynamics_fig = None
        if log_grads_every:
            dynamics_fig = plot_training_dynamics(run.info.run_id, model)
            mlflow.log_figure(dynamics_fig, "training/dynamics.png")

        print("\n===== TEST =====")
        result = evaluate(model, test_loader, classes, device, split="test")

        if log_model:
            # Copie CPU en pickle : rechargeable sans GPU, et embarque la classe du modèle.
            model_cpu = copy.deepcopy(model).cpu().eval()
            example = next(iter(test_loader))[0][:2].cpu().numpy()
            with torch.no_grad():
                example_out = model_cpu(torch.from_numpy(example)).numpy()
            mlflow.pytorch.log_model(
                model_cpu,
                name="model",
                signature=infer_signature(example, example_out),
                serialization_format="pickle",
            )

        print(f"\nRun MLflow : {run.info.run_id}")
        print(f"Visualiser : mlflow ui --backend-store-uri \"{mlflow.get_tracking_uri()}\"")

    result["run_id"] = run.info.run_id
    result["best_val_f1"] = best_f1
    if dynamics_fig is not None:
        result["dynamics_figure"] = dynamics_fig
    return result
