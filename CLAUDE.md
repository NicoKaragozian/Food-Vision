# CLAUDE.md — Food Vision (MIA304 · UdeSA 2026)

## Qué es este proyecto

TP Final de la materia *Visión y Percepción Computarizada* (MIA304). El objetivo es:
> Dada una foto de un plato de comida, identificar la categoría (Food-101) y estimar calorías + macronutrientes (proteínas, carbs, grasas) a partir de un lookup nutricional estático.

Integrantes: Fede · Nico · Benja.

## Entorno

- Python 3.10+, PyTorch + torchvision
- **GPU local con CUDA — no Colab**. Antes de entrenar verificar `torch.cuda.is_available()`.
- EfficientNet-B0 como backbone principal (5.3M parámetros, ~2-3 h de entrenamiento).

## Setup rápido

```bash
cd "TP Final"
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"  # debe ser True
```

## Estructura del código

```
src/
  config.py      ← constantes globales (DEVICE, rutas, clases, hiperparámetros)
  transforms.py  ← get_train/val/inference_transform(), denormalize()
  model.py       ← build_model(), freeze_backbone(), unfreeze_last_blocks(),
                    load_model(), predict(), get_embeddings()
  nutrition.py   ← get_nutrition(), estimate_total_calories()  [sin torch]
  pipeline.py    ← FoodVisionPipeline (interfaz pública)
  trainer.py     ← train_one_epoch(), evaluate(), train_model()
  utils.py       ← show_batch(), plot_training_curves(), plot_confusion_matrix(),
                    GradCAM, overlay_gradcam()

notebooks/
  01_EDA.ipynb         ← exploración Food-101 + preview nutrición
  02_baseline.ipynb    ← head-only training → weights/baseline.pt
  03_finetuning.ipynb  ← fine-tuning diferencial → weights/model_v1.pt
  04_embeddings.ipynb  ← t-SNE + KNN retrieval
  05_evaluation.ipynb  ← métricas finales, Grad-CAM, demo pipeline

data/
  nutrition_lookup.json  ← 101 categorías con kcal/macros por 100g (USDA FoodData Central)
  food-101/              ← NO commitear (5 GB, se descarga con torchvision)

weights/
  baseline.pt   ← se genera al correr notebook 02
  model_v1.pt   ← se genera al correr notebook 03  [NO commitear]
```

## Orden de ejecución de notebooks

```
01_EDA → 02_baseline → 03_finetuning → 04_embeddings
                                     ↘ 05_evaluation
```
Los notebooks 04 y 05 requieren `model_v1.pt`.

## Uso del pipeline desde Python

```python
from src.pipeline import FoodVisionPipeline

pipe = FoodVisionPipeline("weights/model_v1.pt")
result = pipe.analyze("foto.jpg")

# result =
# {
#   "top_prediction": {"class": "pizza", "confidence": 0.92},
#   "alternatives": [{"class": "flatbread", "confidence": 0.05}, ...],
#   "nutrition": {
#     "per_100g": {"calories": 266, "protein_g": 11.0, ...},
#     "estimated_portion": {"portion_g": 250, "calories": 665.0, ...}
#   }
# }
```

## Métricas objetivo

| Métrica          | Mínimo | Ideal  |
|------------------|--------|--------|
| Top-1 accuracy   | > 75%  | > 85%  |
| Top-5 accuracy   | > 90%  | > 95%  |
| Error calórico   | < 20%  | < 10%  |
| Inferencia/img   | < 2 s  | < 500ms|

## Convenciones del proyecto

- Comentarios y docstrings **en español**
- Normalización ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- Backbone principal: `"efficientnet_b0"` — también soporta `"resnet50"` para comparativas
- `src/config.py` es la única fuente de verdad para hiperparámetros y rutas

## Qué NO hacer

- No commitear `data/food-101/` (5 GB) ni `weights/*.pt` (cientos de MB)
- No asumir Colab — el entorno es local con GPU
- No implementar la API FastAPI en el TP (solo el esqueleto en `api/main.py`)
- No entrenar en CPU (no es viable para Food-101)

## Referencia completa

Ver `PLAN.md` para el detalle completo: dataset, arquitecturas, cronograma, papers clave y la fuente del `nutrition_lookup.json`.
