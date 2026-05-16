"""
Genera data/nutrition_foodseg103.json consultando USDA FoodData Central para los
103 ingredientes de FoodSeg103.

Uso:
    export FDC_API_KEY=xxxx
    python3 scripts/build_nutrition_foodseg103.py          # merge-aware (acumula matches)
    python3 scripts/build_nutrition_foodseg103.py --fresh  # empieza de cero

Estrategia:
    1. USDA con CUSTOM_QUERIES_FS103 (queries ajustadas por ingrediente).
    2. Si USDA no devuelve nutrientes completos, fallback a
       data/nutrition_foodseg103_overrides.json (curado manual).
    3. Si tampoco está en overrides y se corre sin --fresh, preservar la entrada
       del build previo (merge-aware).
    4. Reutiliza _search_with_fallback, _extract_nutrients y NUTRIENT_MAP del
       script de Food-101 para no duplicar la lógica de USDA.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.config import FOODSEG103_CLASSES, FOODSEG103_NUTRITION_PATH
from scripts.build_nutrition_lookup import (
    NUTRIENT_MAP,
    _search_with_fallback,
    _extract_nutrients,
)

# Porción por defecto para ingredientes individuales: 100 g (los valores USDA
# ya son por 100 g, así que esta es la unidad natural de reporte).
DEFAULT_PORTION_G = 100

# Queries ajustadas para los ingredientes de FoodSeg103.
# - Algunos nombres del dataset son compuestos ("chicken duck") o asiáticos
#   ("hanamaki baozi") y necesitan una query más amigable para USDA.
# - Otros son ingredientes genéricos crudos y se buscan con "raw"/"cooked" para
#   matchear las entradas estables del SR Legacy.
CUSTOM_QUERIES_FS103: dict = {
    "egg tart":               "egg custard tart",
    "biscuit":                "cookie biscuit",
    "cheese butter":          "cheese cream",
    "wine":                   "wine red",
    "milkshake":              "milkshake vanilla",
    "coffee":                 "coffee brewed",
    "juice":                  "fruit juice orange",
    "milk":                   "milk whole",
    "tea":                    "tea brewed",
    "almond":                 "almonds raw",
    "red beans":              "kidney beans cooked",
    "cashew":                 "cashew nuts raw",
    "dried cranberries":      "cranberries dried sweetened",
    "soy":                    "soybeans cooked",
    "walnut":                 "walnuts raw",
    "peanut":                 "peanuts raw",
    "egg":                    "egg whole cooked",
    "apple":                  "apple raw",
    "date":                   "dates medjool",
    "apricot":                "apricots raw",
    "avocado":                "avocado raw",
    "banana":                 "banana raw",
    "strawberry":             "strawberries raw",
    "cherry":                 "cherries raw sweet",
    "blueberry":              "blueberries raw",
    "raspberry":              "raspberries raw",
    "mango":                  "mango raw",
    "olives":                 "olives green",
    "peach":                  "peach raw",
    "lemon":                  "lemon raw",
    "pear":                   "pear raw",
    "fig":                    "figs raw",
    "pineapple":              "pineapple raw",
    "grape":                  "grapes raw",
    "kiwi":                   "kiwi fruit raw",
    "melon":                  "cantaloupe melon raw",
    "orange":                 "orange raw",
    "watermelon":             "watermelon raw",
    "steak":                  "beef steak grilled",
    "pork":                   "pork cooked",
    "chicken duck":           "chicken roasted",
    "sausage":                "pork sausage cooked",
    "fried meat":             "beef stir fried",
    "lamb":                   "lamb cooked",
    "sauce":                  "soy sauce",
    "crab":                   "crab cooked",
    "fish":                   "fish cooked",
    "shellfish":              "clams cooked",
    "shrimp":                 "shrimp cooked",
    "soup":                   "vegetable soup canned",
    "bread":                  "white bread",
    "corn":                   "corn yellow cooked",
    "hamburg":                "hamburger patty cooked",
    "pizza":                  "pizza cheese",
    " hanamaki baozi":        "steamed bun bao chinese",
    "wonton dumplings":       "wonton soup",
    "pasta":                  "spaghetti cooked",
    "noodles":                "noodles cooked",
    "rice":                   "rice white cooked",
    "pie":                    "apple pie",
    "tofu":                   "tofu firm",
    "eggplant":               "eggplant cooked",
    "potato":                 "potato baked",
    "garlic":                 "garlic raw",
    "cauliflower":            "cauliflower raw",
    "tomato":                 "tomato red raw",
    "kelp":                   "kelp seaweed raw",
    "seaweed":                "nori seaweed dried",
    "spring onion":           "scallions raw",
    "rape":                   "broccoli rabe cooked",
    "ginger":                 "ginger root raw",
    "okra":                   "okra cooked",
    "lettuce":                "lettuce green leaf raw",
    "pumpkin":                "pumpkin raw",
    "cucumber":               "cucumber raw",
    "white radish":           "daikon radish raw",
    "carrot":                 "carrots raw",
    "asparagus":              "asparagus raw",
    "bamboo shoots":          "bamboo shoots cooked",
    "broccoli":               "broccoli raw",
    "celery stick":           "celery raw",
    "cilantro mint":          "cilantro fresh",
    "snow peas":              "snow peas raw",
    " cabbage":               "cabbage raw",
    "bean sprouts":           "mung bean sprouts raw",
    "onion":                  "onions raw",
    "pepper":                 "bell pepper raw",
    "green beans":            "green beans raw",
    "French beans":           "green beans raw",
    "king oyster mushroom":   "king oyster mushroom",
    "shiitake":               "shiitake mushrooms raw",
    "enoki mushroom":         "enoki mushrooms raw",
    "oyster mushroom":        "oyster mushrooms raw",
    "white button mushroom":  "white button mushrooms raw",
    "salad":                  "garden salad mixed greens",
    "other ingredients":      "mixed vegetables cooked",
}


def _query_for(name: str) -> str:
    """Devuelve la query USDA para un nombre de clase, normalizando espacios."""
    name_clean = name.strip()
    return CUSTOM_QUERIES_FS103.get(name, CUSTOM_QUERIES_FS103.get(name_clean, name_clean))


def build(fresh: bool = False) -> None:
    api_key = os.environ.get("FDC_API_KEY", "").strip()
    if not api_key:
        sys.exit("Error: exportá FDC_API_KEY antes de correr el script.\n"
                 "  export FDC_API_KEY=xxxx\n"
                 "  Registro gratuito: https://fdc.nal.usda.gov/api-key-signup")

    overrides_path = ROOT / "data" / "nutrition_foodseg103_overrides.json"
    overrides: dict = {}
    if overrides_path.exists():
        with open(overrides_path, encoding="utf-8") as f:
            raw = json.load(f)
        overrides = {k: v for k, v in raw.items() if not k.startswith("_")}
        logging.info("Overrides cargados: %d entradas", len(overrides))

    existing_lookup: dict = {}
    if not fresh and FOODSEG103_NUTRITION_PATH.exists():
        with open(FOODSEG103_NUTRITION_PATH, encoding="utf-8") as f:
            existing_lookup = json.load(f)
        logging.info("Lookup previo cargado: %d entradas (merge-aware)", len(existing_lookup))

    lookup:    dict = {}
    unmatched = []
    log_lines = []
    from_usda = from_override = from_existing = 0

    required_keys = set(NUTRIENT_MAP.values())
    session = requests.Session()

    for cls in FOODSEG103_CLASSES:
        query = _query_for(cls)
        logging.info("[%s]  query='%s'", cls, query)

        food      = _search_with_fallback(query, api_key, session)
        nutrients = _extract_nutrients(food) if food else {}

        if food and required_keys.issubset(nutrients):
            fdc_id    = food.get("fdcId", "?")
            data_type = food.get("dataType", "?")
            desc      = food.get("description", "")
            lookup[cls] = {
                "calories":  round(nutrients["calories"]),
                "protein_g": round(nutrients["protein_g"], 1),
                "carbs_g":   round(nutrients["carbs_g"],   1),
                "fat_g":     round(nutrients["fat_g"],     1),
                "fiber_g":   round(nutrients["fiber_g"],   1),
                "portion_g": DEFAULT_PORTION_G,
            }
            from_usda += 1
            log_lines.append(
                f"{cls}: fdcId={fdc_id}, dataType={data_type}, "
                f"description='{desc}', query='{query}'"
            )

        elif cls in overrides:
            lookup[cls] = overrides[cls]
            from_override += 1
            logging.warning("  [%s] nutrientes incompletos en USDA → override", cls)
            log_lines.append(f"{cls}: SOURCE=override")

        elif cls in existing_lookup:
            lookup[cls] = existing_lookup[cls]
            from_existing += 1
            logging.info("  [%s] preservado de build previo (USDA no respondió)", cls)
            log_lines.append(f"{cls}: SOURCE=existing")

        else:
            unmatched.append(cls)
            logging.error("  [%s] SIN MATCH en USDA ni en overrides", cls)
            log_lines.append(f"{cls}: SOURCE=UNMATCHED")

        time.sleep(0.25)

    ordered = {cls: lookup[cls] for cls in FOODSEG103_CLASSES if cls in lookup}
    with open(FOODSEG103_NUTRITION_PATH, "w", encoding="utf-8") as f:
        json.dump(ordered, f, indent=2, ensure_ascii=False)

    log_path = ROOT / "build_log_foodseg103.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")

    total = len(FOODSEG103_CLASSES)
    print(f"\n{'='*60}")
    print(f"Lookup FoodSeg103 generado: {len(ordered)}/{total} entradas")
    print(f"  Desde USDA FDC:    {from_usda}")
    print(f"  Desde overrides:   {from_override}")
    if from_existing:
        print(f"  Desde build previo:{from_existing}")
    if unmatched:
        print(f"  SIN MATCH ({len(unmatched)}): {', '.join(unmatched)}")
    print(f"Archivo: {FOODSEG103_NUTRITION_PATH}")
    print(f"Log:     {log_path}")
    print(f"{'='*60}\n")

    if unmatched:
        sys.exit(
            f"ADVERTENCIA: {len(unmatched)} categorías sin datos. "
            f"Agregá entradas en data/nutrition_foodseg103_overrides.json para: {unmatched}"
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Genera data/nutrition_foodseg103.json desde USDA FDC.")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignorar lookup existente y empezar de cero (no merge-aware)",
    )
    args = parser.parse_args()
    build(fresh=args.fresh)
