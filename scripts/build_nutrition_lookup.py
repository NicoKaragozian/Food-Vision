"""
Script build-time: genera data/nutrition_lookup.json consultando USDA FoodData Central.

Uso:
    export FDC_API_KEY=xxxx
    python3 scripts/build_nutrition_lookup.py

Requiere:
    pip install requests>=2.31
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.config import FOOD101_CLASSES, NUTRITION_PATH
from scripts.food101_portions import FOOD101_PORTIONS

FDC_BASE   = "https://api.nal.usda.gov/fdc/v1"
DATA_TYPES = ["Survey (FNDDS)", "SR Legacy"]

# INFOODS nutrient numbers (más estables que nutrientId entre datasets)
NUTRIENT_MAP = {
    "208": "calories",   # Energy (kcal)
    "203": "protein_g",  # Protein
    "205": "carbs_g",    # Carbohydrate, by difference
    "204": "fat_g",      # Total lipid (fat)
    "291": "fiber_g",    # Fiber, total dietary
}

# Queries ajustadas para categorías que no matchean bien con el nombre literal
CUSTOM_QUERIES: dict[str, str] = {
    "baby_back_ribs":        "pork ribs barbecue",
    "beef_carpaccio":        "beef carpaccio raw",
    "beef_tartare":          "beef tartare raw",
    "beet_salad":            "beet salad",
    "bibimbap":              "bibimbap korean rice bowl",
    "breakfast_burrito":     "breakfast burrito egg",
    "cannoli":               "cannoli pastry",
    "caprese_salad":         "caprese salad tomato mozzarella",
    "cheese_plate":          "cheese assorted",
    "chicken_quesadilla":    "quesadilla chicken",
    "chocolate_mousse":      "chocolate mousse dessert",
    "clam_chowder":          "clam chowder soup",
    "crab_cakes":            "crab cake",
    "creme_brulee":          "creme brulee custard",
    "croque_madame":         "croque madame sandwich ham egg",
    "cup_cakes":             "cupcake frosted",
    "deviled_eggs":          "deviled eggs",
    "dumplings":             "dumplings steamed",
    "edamame":               "edamame soybeans",
    "eggs_benedict":         "eggs benedict hollandaise",
    "escargots":             "snails cooked garlic butter",
    "filet_mignon":          "beef tenderloin steak",
    "fish_and_chips":        "fish and chips fried",
    "foie_gras":             "liver pate goose duck",
    "french_onion_soup":     "french onion soup",
    "fried_calamari":        "calamari fried squid",
    "frozen_yogurt":         "frozen yogurt",
    "garlic_bread":          "garlic bread",
    "grilled_cheese_sandwich": "grilled cheese sandwich",
    "grilled_salmon":        "salmon grilled",
    "gyoza":                 "gyoza pot sticker dumpling",
    "hot_and_sour_soup":     "hot and sour soup chinese",
    "hot_dog":               "hot dog frankfurter bun",
    "huevos_rancheros":      "huevos rancheros eggs mexican",
    "ice_cream":             "ice cream",
    "lobster_bisque":        "lobster bisque soup",
    "lobster_roll_sandwich": "lobster roll sandwich",
    "macaroni_and_cheese":   "macaroni and cheese",
    "macarons":              "macaron french almond cookie",
    "miso_soup":             "miso soup japanese",
    "pad_thai":              "pad thai noodles",
    "panna_cotta":           "panna cotta dessert",
    "peking_duck":           "peking duck roasted",
    "pho":                   "pho beef noodle soup vietnamese",
    "pork_chop":             "pork chop cooked",
    "poutine":               "poutine fries cheese gravy",
    "prime_rib":             "prime rib beef roast",
    "pulled_pork_sandwich":  "pulled pork sandwich",
    "ramen":                 "ramen noodle soup",
    "red_velvet_cake":       "red velvet cake",
    "samosa":                "samosa fried pastry",
    "sashimi":               "sashimi raw fish",
    "seaweed_salad":         "seaweed salad",
    "shrimp_and_grits":      "shrimp and grits",
    "spaghetti_bolognese":   "spaghetti with meat sauce bolognese",
    "spaghetti_carbonara":   "spaghetti carbonara",
    "spring_rolls":          "spring rolls fried",
    "strawberry_shortcake":  "strawberry shortcake",
    "takoyaki":              "takoyaki octopus ball japanese",
    "tuna_tartare":          "tuna tartare raw",
}


def _search(query: str, api_key: str, session: requests.Session) -> dict | None:
    """Retorna el primer hit de /foods/search, o None si no hay resultado."""
    params = {
        "query":    query,
        "dataType": DATA_TYPES,
        "pageSize": 5,
        "api_key":  api_key,
    }
    for attempt in range(3):
        try:
            resp = session.get(f"{FDC_BASE}/foods/search", params=params, timeout=10)
            resp.raise_for_status()
            foods = resp.json().get("foods", [])
            return foods[0] if foods else None
        except (requests.RequestException, ValueError) as exc:
            if attempt == 2:
                logging.warning("  error tras 3 intentos: %s", exc)
                return None
            time.sleep(2 ** attempt)
    return None


def _extract_nutrients(food: dict) -> dict:
    """Extrae los 5 macros de la lista foodNutrients, filtrando por nutrientNumber."""
    out = {}
    for fn in food.get("foodNutrients", []):
        num = str(fn.get("nutrientNumber", ""))
        if num in NUTRIENT_MAP:
            out[NUTRIENT_MAP[num]] = float(fn.get("value") or 0.0)
    return out


def build() -> None:
    api_key = os.environ.get("FDC_API_KEY", "").strip()
    if not api_key:
        sys.exit("Error: exportá FDC_API_KEY antes de correr el script.\n"
                 "  export FDC_API_KEY=xxxx\n"
                 "  Registro gratuito: https://fdc.nal.usda.gov/api-key-signup")

    overrides_path = ROOT / "data" / "nutrition_overrides.json"
    overrides: dict = {}
    if overrides_path.exists():
        with open(overrides_path, encoding="utf-8") as f:
            raw = json.load(f)
        # ignorar la clave _comment si existe
        overrides = {k: v for k, v in raw.items() if not k.startswith("_")}
        logging.info("Overrides cargados: %d entradas", len(overrides))

    lookup:    dict[str, dict] = {}
    unmatched: list[str]       = []
    log_lines: list[str]       = []
    from_usda = from_override = 0

    required_keys = set(NUTRIENT_MAP.values())
    session = requests.Session()

    for food_class in FOOD101_CLASSES:
        query = CUSTOM_QUERIES.get(food_class, food_class.replace("_", " "))
        logging.info("[%s]  query='%s'", food_class, query)

        food      = _search(query, api_key, session)
        nutrients = _extract_nutrients(food) if food else {}

        if food and required_keys.issubset(nutrients):
            fdc_id    = food.get("fdcId", "?")
            data_type = food.get("dataType", "?")
            desc      = food.get("description", "")
            portion   = FOOD101_PORTIONS.get(food_class, 200)

            lookup[food_class] = {
                "calories":  round(nutrients["calories"]),
                "protein_g": round(nutrients["protein_g"], 1),
                "carbs_g":   round(nutrients["carbs_g"],   1),
                "fat_g":     round(nutrients["fat_g"],     1),
                "fiber_g":   round(nutrients["fiber_g"],   1),
                "portion_g": portion,
            }
            from_usda += 1
            log_lines.append(
                f"{food_class}: fdcId={fdc_id}, dataType={data_type}, "
                f"description='{desc}', query='{query}'"
            )

        elif food_class in overrides:
            lookup[food_class] = overrides[food_class]
            from_override += 1
            logging.warning("  [%s] nutrientes incompletos en USDA → override", food_class)
            log_lines.append(f"{food_class}: SOURCE=override")

        else:
            unmatched.append(food_class)
            logging.error("  [%s] SIN MATCH en USDA ni en overrides", food_class)
            log_lines.append(f"{food_class}: SOURCE=UNMATCHED")

        time.sleep(0.1)

    # Escribir lookup (orden canónico = FOOD101_CLASSES)
    ordered = {cls: lookup[cls] for cls in FOOD101_CLASSES if cls in lookup}
    with open(NUTRITION_PATH, "w", encoding="utf-8") as f:
        json.dump(ordered, f, indent=2, ensure_ascii=False)

    # Escribir log de auditoría
    log_path = ROOT / "build_log.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")

    total = len(FOOD101_CLASSES)
    print(f"\n{'='*60}")
    print(f"Lookup generado: {len(ordered)}/{total} entradas")
    print(f"  Desde USDA FDC:    {from_usda}")
    print(f"  Desde overrides:   {from_override}")
    if unmatched:
        print(f"  SIN MATCH ({len(unmatched)}): {', '.join(unmatched)}")
    print(f"Archivo: {NUTRITION_PATH}")
    print(f"Log:     {log_path}")
    print(f"{'='*60}\n")

    if unmatched:
        sys.exit(
            f"ADVERTENCIA: {len(unmatched)} categorías sin datos. "
            f"Agregá entradas en data/nutrition_overrides.json para: {unmatched}"
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    build()
