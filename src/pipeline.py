"""
Interfaz pública del sistema Food Vision.

Uso:
    from src.pipeline import FoodVisionPipeline
    pipe = FoodVisionPipeline("weights/model_v1.pt")
    result = pipe.analyze("foto_pizza.jpg")
    # → {"top_prediction": {"class": "pizza", "confidence": 0.92}, ...}
"""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from .config import DEFAULT_BACKBONE, DEVICE, FOOD101_CLASSES, WEIGHTS_DIR
from .model import build_model, load_model, predict
from .nutrition import estimate_total_calories


class FoodVisionPipeline:
    """
    Pipeline de clasificación + estimación nutricional desde una imagen.
    Diseñado para ser reutilizable en notebooks, scripts y APIs.
    """

    def __init__(
        self,
        weights_path: Optional[str | Path] = None,
        backbone: str = DEFAULT_BACKBONE,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            weights_path: Ruta al .pt entrenado. Si None, busca en weights/ automáticamente.
                          Si no hay ningún checkpoint, usa backbone ImageNet puro (útil para tests).
            backbone: "efficientnet_b0" (default) o "resnet50"
            device: Autodetecta CUDA si None.
        """
        self.device = device or DEVICE
        self.backbone = backbone
        self.class_names = FOOD101_CLASSES

        if weights_path is None:
            for candidate in ["model_v1.pt", "baseline.pt"]:
                p = WEIGHTS_DIR / candidate
                if p.exists():
                    weights_path = p
                    break

        if weights_path is not None:
            self.model = load_model(weights_path, backbone=backbone, device=self.device)
            print(f"Modelo cargado: {weights_path}")
        else:
            self.model = build_model(backbone=backbone, pretrained=True)
            self.model.to(self.device).eval()
            print("Aviso: sin pesos entrenados — usando backbone ImageNet puro.")

        print(f"Device: {self.device} | Backbone: {backbone}")

    def analyze(
        self,
        image: "str | Path | Image.Image | np.ndarray",
        top_k: int = 3,
        portion_g: Optional[float] = None,
    ) -> dict:
        """
        Analiza una imagen de comida y devuelve clasificación + nutrición.

        Args:
            image: Ruta al archivo, PIL.Image o ndarray RGB/BGR.
            top_k: Número de predicciones alternativas a incluir.
            portion_g: Gramos para estimar calorías. None → usa porción típica del JSON.

        Returns:
            {
                "top_prediction": {"class": str, "confidence": float},
                "alternatives": [{"class": str, "confidence": float}, ...],
                "nutrition": {
                    "per_100g": {...},
                    "estimated_portion": {"portion_g": float, "calories": float, ...}
                }
            }
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")

        predictions = predict(
            self.model, image, self.class_names,
            device=self.device, top_k=top_k,
        )

        top = predictions[0]
        nutrition = estimate_total_calories(top["class"], portion_g=portion_g)

        return {
            "top_prediction": top,
            "alternatives": predictions[1:],
            "nutrition": nutrition,
        }
