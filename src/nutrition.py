"""
Módulo de nutrición: lookup nutricional por categoría de Food-101.
Completamente desacoplado del modelo de visión.
"""

import json
from functools import lru_cache
from pathlib import Path
from typing import Optional

from .config import NUTRITION_PATH


@lru_cache(maxsize=1)
def _load_db(path: str) -> dict:
    """Carga el JSON una sola vez y lo cachea en memoria."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_nutrition(
    food_class: str,
    path: str | Path = NUTRITION_PATH,
) -> Optional[dict]:
    """
    Devuelve datos nutricionales por 100g para una categoría de Food-101.

    Returns:
        Dict con keys: calories, protein_g, carbs_g, fat_g, fiber_g, portion_g
        None si la categoría no existe en el lookup.
    """
    return _load_db(str(path)).get(food_class)


def estimate_total_calories(
    food_class: str,
    portion_g: Optional[float] = None,
    path: str | Path = NUTRITION_PATH,
) -> Optional[dict]:
    """
    Calcula la nutrición total para una porción (típica o custom).

    Args:
        food_class: Categoría de Food-101 (ej. "pizza", "sushi")
        portion_g: Peso en gramos. Si None, usa el valor típico del JSON.

    Returns:
        {
            "per_100g": {...valores base por 100g...},
            "estimated_portion": {
                "portion_g": float,
                "calories": float,
                "protein_g": float,
                "carbs_g": float,
                "fat_g": float,
                "fiber_g": float,
            }
        }
        None si la categoría no existe.
    """
    nutrition = get_nutrition(food_class, path)
    if nutrition is None:
        return None

    grams = portion_g if portion_g is not None else nutrition["portion_g"]
    factor = grams / 100.0

    return {
        "per_100g": nutrition,
        "estimated_portion": {
            "portion_g": grams,
            "calories": round(nutrition["calories"] * factor, 1),
            "protein_g": round(nutrition["protein_g"] * factor, 1),
            "carbs_g":   round(nutrition["carbs_g"]   * factor, 1),
            "fat_g":     round(nutrition["fat_g"]     * factor, 1),
            "fiber_g":   round(nutrition["fiber_g"]   * factor, 1),
        },
    }
