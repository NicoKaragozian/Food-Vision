"""
Módulo de modelo: construcción, carga, predicción y extracción de embeddings.
"""

from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models

from .config import DEFAULT_BACKBONE, NUM_CLASSES
from .transforms import get_inference_transform


def build_model(
    backbone: str = DEFAULT_BACKBONE,
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
) -> nn.Module:
    """
    Construye un clasificador con transfer learning.

    Args:
        backbone: "efficientnet_b0" (recomendado) o "resnet50"
        num_classes: 101 para Food-101
        pretrained: si cargar pesos de ImageNet

    Returns:
        Modelo con cabeza reemplazada por Dropout(0.3) + Linear(num_classes)
    """
    if backbone == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features  # 1280
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes),
        )
    elif backbone == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features  # 2048
        model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes),
        )
    else:
        raise ValueError(f"Backbone '{backbone}' no soportado. Usar 'efficientnet_b0' o 'resnet50'.")

    return model


def freeze_backbone(model: nn.Module, backbone: str = DEFAULT_BACKBONE) -> None:
    """
    Congela todos los parámetros del backbone; solo la cabeza queda entrenable.
    Usar antes de la Fase 1 (head-only training).
    """
    for param in model.parameters():
        param.requires_grad = False

    head = model.classifier if backbone == "efficientnet_b0" else model.fc
    for param in head.parameters():
        param.requires_grad = True


def unfreeze_last_blocks(
    model: nn.Module,
    n_blocks: int = 3,
    backbone: str = DEFAULT_BACKBONE,
) -> None:
    """
    Descongela los últimos n_blocks bloques del backbone para fine-tuning.
    EfficientNet-B0 tiene 9 bloques en model.features (índices 0-8).
    Dejar n_blocks=3 descongela features[6], features[7], features[8] + classifier.
    """
    for param in model.parameters():
        param.requires_grad = True

    if backbone == "efficientnet_b0":
        n_freeze = len(model.features) - n_blocks  # 9 - 3 = 6
        for i, block in enumerate(model.features):
            if i < n_freeze:
                for param in block.parameters():
                    param.requires_grad = False
    # ResNet50: todos los parámetros quedan libres (comportamiento correcto para fine-tuning)


def get_param_groups(
    model: nn.Module,
    backbone_lr: float,
    head_lr: float,
    backbone: str = DEFAULT_BACKBONE,
) -> list[dict]:
    """
    Retorna param groups con learning rates diferenciales para AdamW.
    Backbone usa lr bajo; cabeza usa lr más alto (mejor convergencia en fine-tuning).
    """
    if backbone == "efficientnet_b0":
        head_params = list(model.classifier.parameters())
        head_ids = {id(p) for p in head_params}
        backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
    else:
        head_params = list(model.fc.parameters())
        head_ids = {id(p) for p in head_params}
        backbone_params = [p for p in model.parameters() if id(p) not in head_ids]

    return [
        {"params": [p for p in backbone_params if p.requires_grad], "lr": backbone_lr},
        {"params": [p for p in head_params if p.requires_grad], "lr": head_lr},
    ]


def load_model(
    weights_path: Union[str, Path],
    backbone: str = DEFAULT_BACKBONE,
    num_classes: int = NUM_CLASSES,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """Carga un checkpoint .pt en un modelo recién inicializado."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(backbone=backbone, num_classes=num_classes, pretrained=False)
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model


def predict(
    model: nn.Module,
    image: Union["Image.Image", np.ndarray],
    class_names: list[str],
    device: Optional[torch.device] = None,
    top_k: int = 3,
) -> list[dict]:
    """
    Clasifica una imagen y devuelve las top_k predicciones.

    Returns:
        [{"class": str, "confidence": float}, ...] ordenado por confianza descendente
    """
    if device is None:
        device = next(model.parameters()).device

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    tensor = get_inference_transform()(image).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]

    top_probs, top_indices = torch.topk(probs, k=top_k)

    return [
        {"class": class_names[idx], "confidence": prob.item()}
        for idx, prob in zip(top_indices.tolist(), top_probs.tolist())
    ]


def get_embeddings(
    model: nn.Module,
    dataloader,
    device: Optional[torch.device] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extrae embeddings del penúltimo layer (avgpool) usando un forward hook.
    No modifica la arquitectura del modelo.

    Útil para t-SNE y KNN retrieval (notebook 04).

    Returns:
        (embeddings, labels): shape (N, 1280) y (N,) para EfficientNet-B0
    """
    if device is None:
        device = next(model.parameters()).device

    activation: dict = {}

    def _hook(module, input, output):
        activation["emb"] = output.detach()

    handle = model.avgpool.register_forward_hook(_hook)

    embeddings_list, labels_list = [], []
    model.eval()

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            _ = model(images)
            emb = activation["emb"].flatten(1).cpu().numpy()
            embeddings_list.append(emb)
            labels_list.append(labels.numpy())

    handle.remove()

    return np.vstack(embeddings_list), np.concatenate(labels_list)
