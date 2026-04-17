"""
Utilidades de visualización, métricas y Grad-CAM.
"""

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .transforms import denormalize


# ── Visualización ──────────────────────────────────────────────────────────────

def show_batch(
    images: torch.Tensor,
    labels: list[str],
    nrow: int = 8,
    figsize: tuple = (16, 4),
):
    """Visualiza un batch de imágenes desnormalizadas con sus etiquetas."""
    imgs = denormalize(images[:nrow]).permute(0, 2, 3, 1).numpy()
    n = len(imgs)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]
    for ax, img, label in zip(axes, imgs, labels[:n]):
        ax.imshow(img)
        ax.set_title(label.replace("_", "\n"), fontsize=7)
        ax.axis("off")
    plt.tight_layout()
    return fig


def plot_training_curves(history: dict, figsize: tuple = (12, 4)):
    """Grafica loss y accuracy (top-1 y top-5) de entrenamiento/validación."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    ax1.plot(epochs, history["train_loss"], "b-o", ms=4, label="Train")
    ax1.plot(epochs, history["val_loss"],   "r-o", ms=4, label="Val")
    ax1.set(xlabel="Época", ylabel="Loss", title="Curva de pérdida")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(epochs, history["train_acc"], "b-o",  ms=4, label="Train Top-1")
    ax2.plot(epochs, history["val_top1"],  "r-o",  ms=4, label="Val Top-1")
    ax2.plot(epochs, history["val_top5"],  "g--o", ms=4, label="Val Top-5")
    ax2.set(xlabel="Época", ylabel="Accuracy", title="Curva de accuracy")
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    figsize: tuple = (24, 22),
):
    """Heatmap de la matriz de confusión 101×101."""
    import seaborn as sns
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm, annot=False, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set(xlabel="Predicho", ylabel="Real", title="Matriz de Confusión — Food-101")
    plt.xticks(rotation=90, fontsize=5)
    plt.yticks(rotation=0, fontsize=5)
    plt.tight_layout()
    return fig


# ── Métricas ───────────────────────────────────────────────────────────────────

def top_k_accuracy(outputs: torch.Tensor, targets: torch.Tensor, k: int = 5) -> float:
    """Calcula top-k accuracy dado logits y etiquetas reales."""
    with torch.no_grad():
        _, pred = outputs.topk(k, dim=1)
        return pred.eq(targets.view(-1, 1).expand_as(pred)).any(dim=1).float().mean().item()


def per_class_accuracy(cm: np.ndarray) -> np.ndarray:
    """Accuracy por clase a partir de la matriz de confusión."""
    return cm.diagonal() / cm.sum(axis=1).clip(min=1)


# ── Grad-CAM ───────────────────────────────────────────────────────────────────

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM).
    Visualiza qué regiones de la imagen activan la predicción del modelo.

    Referencia: Selvaraju et al., ICCV 2017. https://arxiv.org/abs/1610.02391

    Uso:
        cam = GradCAM(model, model.features[-1])  # último bloque de EfficientNet-B0
        heatmap = cam.generate(input_tensor)
        overlay = overlay_gradcam(pil_image, heatmap)
        cam.remove_hooks()
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self._handles = [
            target_layer.register_forward_hook(self._save_activations),
            target_layer.register_full_backward_hook(self._save_gradients),
        ]

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(
        self,
        input_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
    ) -> np.ndarray:
        """
        Genera el mapa de calor para la clase especificada.

        Args:
            input_tensor: Tensor (1, C, H, W) normalizado, mismo device que el modelo.
            class_idx: Clase objetivo. None → predicción top-1.

        Returns:
            Heatmap (H, W) normalizado en [0, 1].
        """
        self.model.eval()
        logits = self.model(input_tensor)

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        self.model.zero_grad()
        logits[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    def remove_hooks(self):
        for h in self._handles:
            h.remove()


def overlay_gradcam(
    image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    size: int = 224,
) -> Image.Image:
    """
    Superpone el heatmap Grad-CAM sobre la imagen original con colormap JET.

    Args:
        alpha: Peso del heatmap en la mezcla (0 = solo imagen, 1 = solo heatmap).
    """
    import cv2
    img_arr = np.array(image.resize((size, size)))
    colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    overlay = np.uint8(alpha * colored + (1 - alpha) * img_arr)
    return Image.fromarray(overlay)
