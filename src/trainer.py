"""
Loop de entrenamiento reutilizable para notebooks 02 (baseline) y 03 (fine-tuning).
"""

from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
from tqdm import tqdm


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    Una época de entrenamiento.

    Returns:
        (avg_loss, top1_accuracy)
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(loader, desc="  train", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """
    Evaluación completa sobre un DataLoader.

    Returns:
        {"loss": float, "top1": float, "top5": float}
    """
    model.eval()
    total_loss, correct1, correct5, total = 0.0, 0, 0, 0

    for images, labels in tqdm(loader, desc="  eval ", leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)

        total_loss += criterion(outputs, labels).item() * images.size(0)
        correct1 += (outputs.argmax(1) == labels).sum().item()

        _, top5_pred = outputs.topk(5, dim=1)
        correct5 += top5_pred.eq(labels.view(-1, 1).expand_as(top5_pred)).any(1).sum().item()
        total += images.size(0)

    return {
        "loss":  total_loss / total,
        "top1":  correct1  / total,
        "top5":  correct5  / total,
    }


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    epochs: int,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    save_path: Optional[Union[str, Path]] = None,
    scheduler=None,
    patience: int = 5,
) -> dict:
    """
    Entrenamiento completo con early stopping y guardado del mejor checkpoint.

    Args:
        save_path: Ruta .pt para guardar el mejor modelo. None = no guardar.
        patience: Épocas consecutivas sin mejora en val_top1 para detener el entrenamiento.

    Returns:
        history: dict con listas train_loss, train_acc, val_loss, val_top1, val_top5
    """
    history: dict = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_top1": [], "val_top5": [],
    }

    best_val_top1 = 0.0
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        print(f"\nÉpoca {epoch}/{epochs}")

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        if scheduler is not None:
            scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_metrics["loss"])
        history["val_top1"].append(val_metrics["top1"])
        history["val_top5"].append(val_metrics["top5"])

        print(
            f"  train loss={train_loss:.4f}  acc={train_acc:.3f}  |  "
            f"val loss={val_metrics['loss']:.4f}  top1={val_metrics['top1']:.3f}  "
            f"top5={val_metrics['top5']:.3f}"
        )

        if val_metrics["top1"] > best_val_top1:
            best_val_top1 = val_metrics["top1"]
            epochs_no_improve = 0
            if save_path is not None:
                torch.save(model.state_dict(), save_path)
                print(f"  ✓ Guardado en {save_path}  (top1={best_val_top1:.3f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\n  Early stopping: {patience} épocas sin mejora.")
                break

    print(f"\nEntrenamiento finalizado. Mejor val_top1: {best_val_top1:.3f}")
    return history
