"""
Configuración centralizada del proyecto Food Vision.
Importar con: from src.config import DEVICE, NUM_CLASSES, ...
"""

from pathlib import Path
import torch

# ── Rutas ──────────────────────────────────────────────────────────────────────
ROOT_DIR       = Path(__file__).parent.parent
DATA_DIR       = ROOT_DIR / "data"
WEIGHTS_DIR    = ROOT_DIR / "weights"
FOOD101_DIR    = DATA_DIR / "food-101"
NUTRITION_PATH = DATA_DIR / "nutrition_lookup.json"

# ── Dataset ────────────────────────────────────────────────────────────────────
NUM_CLASSES = 101

# Clases de Food-101 en orden alfabético (mismo que torchvision.datasets.Food101)
FOOD101_CLASSES = [
    "apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio", "beef_tartare",
    "beet_salad", "beignets", "bibimbap", "bread_pudding", "breakfast_burrito",
    "bruschetta", "caesar_salad", "cannoli", "caprese_salad", "carrot_cake",
    "ceviche", "cheesecake", "cheese_plate", "chicken_curry", "chicken_quesadilla",
    "chicken_wings", "chocolate_cake", "chocolate_mousse", "churros", "clam_chowder",
    "club_sandwich", "crab_cakes", "creme_brulee", "croque_madame", "cup_cakes",
    "deviled_eggs", "donuts", "dumplings", "edamame", "eggs_benedict",
    "escargots", "falafel", "filet_mignon", "fish_and_chips", "foie_gras",
    "french_fries", "french_onion_soup", "french_toast", "fried_calamari", "fried_rice",
    "frozen_yogurt", "garlic_bread", "gnocchi", "greek_salad", "grilled_cheese_sandwich",
    "grilled_salmon", "guacamole", "gyoza", "hamburger", "hot_and_sour_soup",
    "hot_dog", "huevos_rancheros", "hummus", "ice_cream", "lasagna",
    "lobster_bisque", "lobster_roll_sandwich", "macaroni_and_cheese", "macarons",
    "miso_soup", "mussels", "nachos", "omelette", "onion_rings", "oysters",
    "pad_thai", "paella", "pancakes", "panna_cotta", "peking_duck",
    "pho", "pizza", "pork_chop", "poutine", "prime_rib",
    "pulled_pork_sandwich", "ramen", "ravioli", "red_velvet_cake", "risotto",
    "samosa", "sashimi", "scallops", "seaweed_salad", "shrimp_and_grits",
    "spaghetti_bolognese", "spaghetti_carbonara", "spring_rolls", "steak", "strawberry_shortcake",
    "sushi", "tacos", "takoyaki", "tiramisu", "tuna_tartare", "waffles",
]

# ── Modelo ─────────────────────────────────────────────────────────────────────
DEFAULT_BACKBONE = "efficientnet_b0"
INPUT_SIZE = 224

# ── Entrenamiento ──────────────────────────────────────────────────────────────
BATCH_SIZE = 64
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fase 1 — head only
BASELINE_LR     = 1e-3
BASELINE_EPOCHS = 5

# Fase 2 — fine-tuning diferencial
FINETUNE_BACKBONE_LR  = 1e-4
FINETUNE_HEAD_LR      = 1e-3
FINETUNE_WEIGHT_DECAY = 1e-4
FINETUNE_EPOCHS       = 25
EARLY_STOP_PATIENCE   = 5
LABEL_SMOOTHING       = 0.1
