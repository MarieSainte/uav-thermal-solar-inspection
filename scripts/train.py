"""Training loop, test evaluation and MLflow logging shared by the CNN notebooks."""

import copy

import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import numpy as np
import torch
from mlflow.models import infer_signature
from sklearn.metrics import f1_score

from evaluation import (compare_runs, evaluate_predictions, log_metrics_if_active,  # noqa: F401
                        log_splits, setup_mlflow)

# Ordinal blue ramp: light = input layer, dark = output layer.
_DEPTH_RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
               "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
_INK, _MUTED, _GRID = "#52514e", "#898781", "#e1e0d9"


def predict(model, loader, device, criterion=None):
    """Return labels, predictions, probabilities and mean loss (None without criterion)."""
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


@torch.no_grad()
def grad_norms(model):
    """L2 norm of each parameter's gradient plus the global norm, measured before clipping."""
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
    """||Δw|| / ||w|| per parameter across one optimizer step (healthy order of magnitude: 1e-3)."""
    names, ratios = [], []
    for n, p in model.named_parameters():
        if n in before:
            names.append(n)
            ratios.append((p - before[n]).norm() / (before[n].norm() + 1e-8))
    return {f"update_ratio/{n}": v for n, v in zip(names, torch.stack(ratios).tolist())}


def plot_training_dynamics(run_id, model):
    """Gradient norms and update ratios of a run, one line per conv / linear kernel."""
    client = mlflow.MlflowClient()
    run = client.get_run(run_id)
    keys = set(run.data.metrics)
    steps_per_epoch = float(run.data.params.get("steps_per_epoch", 1))

    layers = [n for n, p in model.named_parameters() if p.ndim > 1 and f"grad/{n}" in keys]
    if not layers:
        raise ValueError("No grad/* metrics in this run (log_grads_every=0?)")

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

    panels = [(ax_g, "Gradient norm per layer", True),
              (ax_u, "Update ratio ||Δw|| / ||w||", True),
              (ax_t, "Global gradient norm (before clipping)", True),
              (ax_l, "Batch loss", False)]
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
               fontsize=8, title="layer (input → output)", title_fontsize=8)
    fig.tight_layout()
    # Closed so it is not displayed twice; the notebook shows it from the result.
    plt.close(fig)
    return fig


def train(model, criterion, optimizer, train_loader, val_loader=None, num_epochs=30,
          device="cpu", log_grads_every=50, grad_clip=None):
    """Train, then restore the weights with the best validation macro F1.

    Every `log_grads_every` steps (0 / None to disable), log gradient norms, update ratios
    and the batch loss. `grad_clip` is the max gradient norm, None for no clipping.
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

        log_metrics_if_active(epoch_metrics, step=epoch)
        print(msg)

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\nBest model restored (epoch {best_epoch+1}, val_f1_macro = {best_f1:.4f})")
        log_metrics_if_active({"best_val_f1_macro": best_f1, "best_epoch": best_epoch})

    return best_f1


def evaluate(model, loader, classes, device, split="test"):
    """Evaluate a model on a non-shuffled loader whose dataset exposes `.dataframe`."""
    labels, _, probs, _ = predict(model, loader, device)
    dataframe = getattr(loader.dataset, "dataframe", None)
    return evaluate_predictions(labels, probs, classes, split=split, dataframe=dataframe)


def run_experiment(model, criterion, optimizer, train_loader, val_loader, test_loader, classes,
                   experiment_name, run_name=None, num_epochs=15, device=None,
                   params=None, tags=None, log_grads_every=50, grad_clip=None):
    """One MLflow run: training, test evaluation of the best model, model and split logging.

    `params` / `tags` are added to the automatic ones. Returns the `evaluate` dict plus
    `run_id`, `best_val_f1` and `dynamics_figure` (when gradients are logged).
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
            "batch_size_train": train_loader.batch_size,
            "batch_size_eval": test_loader.batch_size,
            "num_epochs": num_epochs,
            "steps_per_epoch": len(train_loader),
            "grad_clip": grad_clip if grad_clip is not None else "none",
            "log_grads_every": log_grads_every or 0,
            "n_train": len(train_loader.dataset),
            "n_val": len(val_loader.dataset),
            "n_test": len(test_loader.dataset),
            "device": str(device),
        }
        mlflow.log_params({**auto_params, **(params or {})})
        mlflow.set_tags(tags or {})
        log_splits()

        best_f1 = train(model, criterion, optimizer, train_loader, val_loader,
                        num_epochs=num_epochs, device=device,
                        log_grads_every=log_grads_every, grad_clip=grad_clip)

        dynamics_fig = None
        if log_grads_every:
            dynamics_fig = plot_training_dynamics(run.info.run_id, model)
            mlflow.log_figure(dynamics_fig, "training/dynamics.png")

        print("\n===== TEST =====")
        result = evaluate(model, test_loader, classes, device, split="test")

        # CPU pickle: reloadable without a GPU and bundles the model class.
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

        print(f"\nMLflow run: {run.info.run_id}")
        print(f"View with: mlflow ui --backend-store-uri \"{mlflow.get_tracking_uri()}\"")

    result["run_id"] = run.info.run_id
    result["best_val_f1"] = best_f1
    if dynamics_fig is not None:
        result["dynamics_figure"] = dynamics_fig
    return result
