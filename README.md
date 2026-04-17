# Food Vision

**Estimación de calorías y macronutrientes desde una imagen de comida.**  
TP Final · MIA304 Visión y Percepción Computarizada · UdeSA 2026

---

## Setup

```bash
# 1. Clonar / entrar al repositorio
cd "TP Final"

# 2. Crear entorno virtual
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Verificar GPU
python3 -c "import torch; print('MPS:', torch.backends.mps.is_available())"   # Mac Apple Silicon
python3 -c "import torch; print('CUDA:', torch.cuda.is_available())"          # NVIDIA
```

> El entrenamiento requiere GPU. En Mac Apple Silicon (MPS) el baseline tarda ~30 min y el fine-tuning ~3-4 horas.

---

## Lookup nutricional

`data/nutrition_lookup.json` se genera desde USDA FoodData Central. La API es gratuita:

```bash
# Registro en: https://fdc.nal.usda.gov/api-key-signup
export FDC_API_KEY=tu_key_aqui

# Primera corrida (desde cero)
python3 scripts/build_nutrition_lookup.py --fresh

# Si quedaron categorías sin match, re-correr (merge-aware, nunca regresa)
python3 scripts/build_nutrition_lookup.py
```

El script tarda ~60-90 s. Si la API de USDA falla con errores 400 transitorios en algunas categorías, simplemente volvé a correr sin `--fresh` — acumula los matches sin pisar los anteriores. El archivo `data/nutrition_overrides.json` contiene los fallbacks manuales para categorías que USDA no cubre.

---

## Orden de ejecución

```
notebooks/01_EDA.ipynb         → Exploración de Food-101 (descarga automática ~5 GB)
notebooks/02_baseline.ipynb    → Transfer learning head-only → weights/baseline.pt
notebooks/03_finetuning.ipynb  → Fine-tuning completo → weights/model_v1.pt
notebooks/04_embeddings.ipynb  → t-SNE + KNN retrieval  (requiere model_v1.pt)
notebooks/05_evaluation.ipynb  → Métricas finales + Grad-CAM + demo  (requiere model_v1.pt)
```

---

## Uso del pipeline

```python
from src.pipeline import FoodVisionPipeline

pipe = FoodVisionPipeline("weights/model_v1.pt")
result = pipe.analyze("foto_pizza.jpg")

print(result["top_prediction"])
# {"class": "pizza", "confidence": 0.92}

print(result["nutrition"]["estimated_portion"])
# {"portion_g": 250, "calories": 665.0, "protein_g": 27.5, "carbs_g": 82.5, ...}
```

---

## Estructura

```
src/           Módulos reutilizables (modelo, nutrición, pipeline, trainer, utils)
notebooks/     Evidencia académica (EDA → entrenamiento → embeddings → evaluación)
data/          nutrition_lookup.json + nutrition_overrides.json + food-101/ (generado al correr 01_EDA)
weights/       Checkpoints entrenados (no se suben a git)
api/           Esqueleto FastAPI (referencia futura, no parte del TP)
scripts/       build_nutrition_lookup.py + food101_portions.py
```

---

## Métricas objetivo

| Métrica        | Mínimo | Ideal  |
|----------------|--------|--------|
| Top-1 accuracy | > 75%  | > 85%  |
| Top-5 accuracy | > 90%  | > 95%  |
| Error calórico | < 20%  | < 10%  |

---

## Referencias

- Bossard et al. (2014) — [Food-101, ECCV](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)
- Meyers et al. (2015) — [Im2Calories, ICCV](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Meyers_Im2Calories_Towards_an_ICCV_2015_paper.pdf)
- Selvaraju et al. (2017) — [Grad-CAM, ICCV](https://arxiv.org/abs/1610.02391)
