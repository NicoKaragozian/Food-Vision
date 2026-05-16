"""
Configuración centralizada del proyecto Food Vision.
Importar con: from src.config import DEVICE, NUM_CLASSES, ...
"""

from pathlib import Path
import torch

# ── Rutas ──────────────────────────────────────────────────────────────────────
ROOT_DIR                  = Path(__file__).parent.parent
DATA_DIR                  = ROOT_DIR / "data"
WEIGHTS_DIR               = ROOT_DIR / "weights"
FOOD101_DIR               = DATA_DIR / "food-101"
NUTRITION_PATH            = DATA_DIR / "nutrition_lookup.json"
FOODSEG103_NUTRITION_PATH = DATA_DIR / "nutrition_foodseg103.json"

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

# ── Dataset: FoodSeg103 ────────────────────────────────────────────────────────
# 103 ingredientes individuales en orden de id (id 1 = "candy", id 103 = "other
# ingredientes"; id 0 = background, omitido). Fuente: HuggingFace dataset
# EduardoPacheco/FoodSeg103, archivo id2label.json.
# Algunos nombres traen artefactos del dataset original (espacios iniciales,
# pares de palabras como "chicken duck") y se preservan tal cual para mantener
# la correspondencia con los ids de las máscaras.
FOODSEG103_CLASSES = [
    "candy", "egg tart", "french fries", "chocolate", "biscuit",
    "popcorn", "pudding", "ice cream", "cheese butter", "cake",
    "wine", "milkshake", "coffee", "juice", "milk",
    "tea", "almond", "red beans", "cashew", "dried cranberries",
    "soy", "walnut", "peanut", "egg", "apple",
    "date", "apricot", "avocado", "banana", "strawberry",
    "cherry", "blueberry", "raspberry", "mango", "olives",
    "peach", "lemon", "pear", "fig", "pineapple",
    "grape", "kiwi", "melon", "orange", "watermelon",
    "steak", "pork", "chicken duck", "sausage", "fried meat",
    "lamb", "sauce", "crab", "fish", "shellfish",
    "shrimp", "soup", "bread", "corn", "hamburg",
    "pizza", " hanamaki baozi", "wonton dumplings", "pasta", "noodles",
    "rice", "pie", "tofu", "eggplant", "potato",
    "garlic", "cauliflower", "tomato", "kelp", "seaweed",
    "spring onion", "rape", "ginger", "okra", "lettuce",
    "pumpkin", "cucumber", "white radish", "carrot", "asparagus",
    "bamboo shoots", "broccoli", "celery stick", "cilantro mint", "snow peas",
    " cabbage", "bean sprouts", "onion", "pepper", "green beans",
    "French beans", "king oyster mushroom", "shiitake", "enoki mushroom", "oyster mushroom",
    "white button mushroom", "salad", "other ingredients",
]
assert len(FOODSEG103_CLASSES) == 103, f"Esperaba 103 clases, hay {len(FOODSEG103_CLASSES)}"

# ── Modelo ─────────────────────────────────────────────────────────────────────
DEFAULT_BACKBONE = "efficientnet_b0"
INPUT_SIZE = 224

# ── Entrenamiento ──────────────────────────────────────────────────────────────
BATCH_SIZE = 64
SEED = 42
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

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
