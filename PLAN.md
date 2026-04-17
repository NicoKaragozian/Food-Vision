# Food Vision — Guía de Implementación para Claude Code

> **Materia:** MIA304 — Visión y Percepción Computarizada · UdeSA 2026  
> **Proyecto:** Estimación de calorías y macronutrientes a partir de imágenes de comida  
> **Este archivo** es una guía completa para que Claude Code implemente el proyecto desde cero. Todo el entorno es **local** (no Colab).

---

## 1. Descripción del Problema

Dado que el monitoreo nutricional manual es tedioso e impreciso, el objetivo es construir un sistema que, a partir de una **fotografía de un plato de comida**, sea capaz de:

1. **Identificar** el alimento presente en la imagen
2. **Estimar las calorías** totales del plato
3. **Desglosar macronutrientes** (proteínas, carbohidratos, grasas) en base a una porción típica

### Pipeline general

```
Imagen de comida
       ↓
  Preprocesamiento (resize, normalización, augmentation)
       ↓
  Modelo de Clasificación CNN (EfficientNet-B0 recomendado, o ResNet50)
  Fine-tuning sobre Food-101
  Salida: clase alimentaria (top-1 / top-5 + confianza)
       ↓
  Lookup Nutricional (JSON estático derivado de USDA FoodData Central)
  Mapping: clase → {calorías, proteínas, carbs, grasas} por 100g
       ↓
  Resultado: {categoria, confianza, alternativas, nutricion_por_100g, porcion_tipica_g, kcal_estimadas}
```

**Alcance y simplificaciones:**
- Un alimento por imagen (no bandejas con múltiples componentes)
- Estimaciones nutricionales promedio por categoría (no estimación exacta de porciones)
- Las calorías estimadas se calculan asumiendo una porción típica estándar por categoría

---

## 2. Stack Tecnológico

| Componente             | Herramienta                         |
|------------------------|-------------------------------------|
| Lenguaje               | Python 3.10+                        |
| Framework DL           | PyTorch + torchvision               |
| Procesamiento imágenes | PIL (Pillow) + OpenCV               |
| Visualización          | Matplotlib + Seaborn                |
| Dataset principal      | Food-101 (clasificación)            |
| Datos nutricionales    | JSON estático (generado desde USDA) |
| Entorno                | Local con GPU NVIDIA (CUDA)         |
| Notebooks              | Jupyter Lab / Jupyter Notebook      |
| Seguimiento (opcional) | Weights & Biases                    |

### Setup del entorno local

1. Crear un entorno virtual con `venv` o `conda`.
2. Instalar PyTorch con soporte CUDA. Antes de instalar, verificar la versión de CUDA con `nvidia-smi` y usar el selector oficial en https://pytorch.org/get-started/locally/ para obtener el comando exacto.
3. Instalar el resto de dependencias desde `requirements.txt`.
4. Verificar que `torch.cuda.is_available()` devuelve `True` antes de empezar.

> **Nota sobre GPU:** EfficientNet-B0 sobre Food-101 tarda ~2-3 horas en una GPU moderna (RTX 3060 o superior). Entrenar en CPU no es viable para este dataset.

---

## 3. Datasets

### 3.1 Food-101 — Dataset Principal

**URL oficial:** https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/  
**Kaggle:** https://www.kaggle.com/dansbecker/food-101  
**Licencia:** CC BY (uso académico libre)  
**Tamaño:** ~5 GB

**Características:**
- 101,000 imágenes de 101 categorías de comida
- 750 imágenes de train + 250 de test por clase (perfectamente balanceado)
- Las imágenes de train tienen algo de ruido intencional (para realismo)
- Benchmark estándar en food recognition desde 2014
- Modelos modernos superan el 92% de top-1 accuracy

**Cómo descargar:** la forma más sencilla es usar `torchvision.datasets.Food101` con `download=True`, que descarga y descomprime automáticamente en la carpeta `data/`. Alternativamente se puede descargar el `.tar.gz` directamente desde la URL oficial y descomprimirlo manualmente.

**Estructura resultante tras la descarga:**
```
data/food-101/
├── images/
│   ├── apple_pie/       (1000 imágenes .jpg)
│   ├── beef_carpaccio/
│   ├── ...              (101 carpetas en total)
│   └── waffles/
├── meta/
│   ├── classes.txt      (lista de las 101 clases)
│   ├── train.txt        (75,750 paths de train)
│   └── test.txt         (25,250 paths de test)
└── README.txt
```

### 3.2 Datos Nutricionales — JSON Estático

Food-101 **no tiene datos nutricionales**. La estrategia es usar un archivo `data/nutrition_lookup.json` pre-armado que mapea cada categoría a sus valores por 100g (fuente: USDA FoodData Central). Este archivo es **estático** — no hace falta llamar a ninguna API durante el entrenamiento ni la inferencia.

**URL de la API USDA (solo si se quiere regenerar el JSON):** https://fdc.nal.usda.gov/  
**API Key gratuita:** https://fdc.nal.usda.gov/api-key-signup.html  
**Documentación:** https://app.swaggerhub.com/apis/fdcnal/food-data_central_api/1.0.1

El JSON tiene la siguiente forma para cada una de las 101 categorías:

```json
{
  "pizza":      {"calories": 266, "protein_g": 11.0, "carbs_g": 33.0, "fat_g": 10.0, "fiber_g": 2.3, "portion_g": 250},
  "hamburger":  {"calories": 295, "protein_g": 17.0, "carbs_g": 24.0, "fat_g": 14.0, "fiber_g": 1.2, "portion_g": 250},
  "sushi":      {"calories": 143, "protein_g": 6.0,  "carbs_g": 24.0, "fat_g": 3.0,  "fiber_g": 0.8, "portion_g": 200}
}
```

El campo `portion_g` indica la porción típica asumida para calcular las calorías totales del plato. Los valores nutricionales completos para las 101 categorías están al final de este documento (sección 8).

---

## 4. Estructura del Proyecto

```
food-vision/
├── notebooks/
│   ├── 01_EDA.ipynb               ← Exploración del dataset Food-101
│   ├── 02_baseline.ipynb          ← Baseline con transfer learning (head only)
│   ├── 03_finetuning.ipynb        ← Fine-tuning progresivo completo
│   ├── 04_embeddings.ipynb        ← Embeddings visuales + retrieval + t-SNE
│   └── 05_evaluation.ipynb        ← Evaluación final y demo del pipeline
├── src/
│   ├── __init__.py
│   ├── model.py                   ← build_model(), load_model(), predict(), get_embeddings()
│   ├── nutrition.py               ← get_nutrition(), estimate_total_calories()
│   └── pipeline.py                ← init(), analyze() — interfaz pública del sistema
├── data/
│   ├── food-101/                  ← Dataset descargado (generado al correr EDA)
│   └── nutrition_lookup.json      ← Mapa: categoría → nutrición por 100g (ver sección 8)
├── weights/
│   ├── baseline.pt                ← Checkpoint fase 1 (se genera al entrenar)
│   └── model_v1.pt                ← Mejor modelo (se genera al entrenar)
├── api/                           ← Esqueleto para uso futuro — no implementar en el TP
│   └── main.py                    ← FastAPI ~30 líneas encima del pipeline
├── requirements.txt
└── README.md
```

---

## 5. Módulos `src/` — Qué debe hacer cada uno

### `src/model.py`

Responsabilidades:
- `build_model(backbone, num_classes=101)`: construye el modelo con transfer learning. Backbone por defecto: EfficientNet-B0 con pesos ImageNet. Reemplaza la última capa por una de 101 neuronas.
- `load_model(weights_path, backbone)`: carga un modelo ya entrenado desde `.pt`.
- `predict(image_pil, model, top_k=3)`: recibe una imagen PIL, devuelve lista de `{categoria, confianza}` ordenada por confianza descendente.
- `get_embeddings(image_pil, model)`: extrae el embedding de la penúltima capa (antes del classifier), normalizado L2. Útil para retrieval.

Las transformaciones de imagen deben usar la normalización estándar de ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]). El train transform debe incluir data augmentation (RandomResizedCrop, HorizontalFlip, ColorJitter, RandomRotation). El val/test transform: Resize(256) → CenterCrop(224).

### `src/nutrition.py`

Responsabilidades:
- Carga `data/nutrition_lookup.json` una sola vez al importar.
- `get_nutrition(categoria)`: devuelve el dict nutricional por 100g, o `None` si no existe.
- `estimate_total_calories(categoria, portion_g=None)`: calcula calorías totales para la porción. Si `portion_g` es `None`, usa el `portion_g` del JSON.

Este módulo no sabe que existe un modelo. Está completamente desacoplado de la parte de visión.

### `src/pipeline.py`

Responsabilidades:
- `init(weights_path, backbone)`: carga el modelo una sola vez. Detecta CUDA automáticamente.
- `analyze(image_pil, portion_g=None)`: función pública principal. Llama a `predict()` y luego a `get_nutrition()`. Devuelve un dict con: `categoria`, `confianza`, `alternativas` (top-2), `nutricion_por_100g`, `porcion_g`, `calorias_estimadas`.

---

## 6. Notebooks — Objetivos por Notebook

### `01_EDA.ipynb` — Exploración del Dataset

1. Descargar Food-101 con `torchvision.datasets.Food101(download=True)`
2. Verificar distribución de clases (debe ser uniforme: 750 train / 250 test)
3. Grid visual de ejemplos: 5 clases aleatorias × 5 imágenes cada una
4. Histograma de resoluciones de imagen
5. Preview del lookup nutricional: tabla con las 101 categorías, calorías y porción típica

### `02_baseline.ipynb` — Baseline (Head Only)

1. Construir modelo con `build_model("efficientnet_b0")`
2. Congelar todas las capas excepto el classifier final
3. Entrenar solo la cabeza, ~5 epochs, lr=1e-3, optimizer Adam
4. Evaluar top-1 y top-5 accuracy en el test set
5. Guardar checkpoint como `weights/baseline.pt`

Meta de este notebook: entender el punto de partida. Se espera ~60-70% top-1.

### `03_finetuning.ipynb` — Fine-tuning Completo

1. Cargar `baseline.pt`
2. Descongelar gradualmente capas del backbone de atrás hacia adelante (en EfficientNet-B0: descongelar los últimos 3 bloques de `features` + el classifier)
3. Usar learning rates diferenciales: lr más bajo para capas del backbone (~1e-4), lr más alto para el classifier (~1e-3)
4. Optimizer: AdamW con weight_decay
5. Scheduler: CosineAnnealingLR o OneCycleLR
6. Early stopping con paciencia de 5 epochs
7. Guardar el mejor modelo como `weights/model_v1.pt`

Meta: top-1 > 80%, top-5 > 92%.

### `04_embeddings.ipynb` — Embeddings y Retrieval

1. Cargar `model_v1.pt`
2. Extraer embeddings de todo el test set usando `get_embeddings()` (usar un forward hook sobre `avgpool`)
3. Visualizar el espacio latente con t-SNE (muestra de ~2000 imágenes, colorear por categoría)
4. Construir índice KNN con similitud coseno (scikit-learn `NearestNeighbors`)
5. Demo de retrieval: dada una imagen nueva, encontrar los 5 más similares y promediar su nutrición
6. Comparar: estimación por hard classification vs. soft retrieval (promedio de vecinos) — ¿cuál tiene menor error calórico promedio?

### `05_evaluation.ipynb` — Evaluación Final y Demo

1. Métricas finales: top-1 y top-5 accuracy en el test set completo
2. Matriz de confusión: top-20 categorías con más errores
3. Curva accuracy vs. umbral de confianza (si confianza < X, reportar "no sé")
4. Ejemplos visuales: 10 predicciones correctas y 10 incorrectas
5. Demo end-to-end del pipeline: cargar una imagen arbitraria → mostrar categoría + confianza + tabla nutricional + calorías estimadas
6. Análisis del error calórico: para cada imagen del test, calcular `|kcal_predichas - kcal_reales|/kcal_reales`

---

## 7. Métricas de Éxito

| Métrica                         | Mínimo  | Ideal   |
|---------------------------------|---------|---------|
| Top-1 Accuracy (test set)       | > 75%   | > 85%   |
| Top-5 Accuracy (test set)       | > 90%   | > 95%   |
| Error calórico promedio         | < 20%   | < 10%   |
| Tiempo de inferencia por imagen | < 2 seg | < 500ms |

---

## 8. Arquitecturas y Backbones

### Opción recomendada: EfficientNet-B0

- **Parámetros:** 5.3M — liviano, entrena rápido
- **Top-1 en ImageNet:** 77.7%
- **Top-1 alcanzable en Food-101 con fine-tuning:** ~85-88%
- **Tiempo de entrenamiento local (GPU moderna, 20 epochs):** ~2-3 horas
- **Torchvision:** `models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)`
- La capa a reemplazar es `model.classifier[1]` (in_features=1280)

### Alternativa: ResNet50

- **Parámetros:** 25.6M
- **Top-1 alcanzable en Food-101:** ~82-85%
- **Más lento de entrenar** que EfficientNet-B0
- La capa a reemplazar es `model.fc` (in_features=2048)
- Ventaja: más citado en la literatura, fácil comparar resultados

### Extensión moderna: ViT (Vision Transformer)

- Instalar con `pip install timm` → `timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=101)`
- Top-1 alcanzable en Food-101: >90%
- Requiere más VRAM — solo intentarlo si la GPU tiene ≥8GB
- Mencionarlo en el informe como comparación con el estado del arte

### Extensión con CLIP (OpenAI)

- `pip install git+https://github.com/openai/CLIP.git`
- Permite zero-shot classification: `"a photo of {clase}"` como prompt de texto
- Útil como baseline sin fine-tuning para comparar en el informe

---

## 9. Extensiones Opcionales (si hay tiempo)

### Detección multi-alimento con YOLOv8

Si el plato tiene múltiples alimentos, en lugar de clasificación simple usar detección. Instalar `ultralytics` → fine-tunear YOLOv8n sobre UEC-Food 256 (tiene bounding boxes). La estrategia: detectar cada región → clasificar con el modelo Food-101 → sumar calorías por componente. Dataset: https://www.kaggle.com/datasets/rkuo2000/uecfood256

### Segmentación de porciones

Usar SAM (Segment Anything Model) para segmentar cada componente del plato, usar el área relativa de cada región como proxy del volumen, y ajustar las calorías por proporción.

### API con FastAPI

El esqueleto de `api/main.py` expone un endpoint `POST /predict` que recibe una imagen y devuelve el JSON nutricional. Se apoya directamente en `src/pipeline.py`, son ~30 líneas. No implementar en el TP, pero dejar el archivo listo como esqueleto comentado.

---

## 10. Checklist de Implementación

### Fase 1 — Setup y EDA (para S2, 02/05)
- [ ] Crear estructura de carpetas del proyecto
- [ ] Crear `requirements.txt` e instalar dependencias
- [ ] Crear `data/nutrition_lookup.json` (ver sección 11)
- [ ] Implementar `src/model.py`, `src/nutrition.py`, `src/pipeline.py`
- [ ] Notebook `01_EDA.ipynb`: descargar Food-101 y visualizar datos
- [ ] Verificar que el pipeline básico funciona end-to-end con una imagen de prueba

### Fase 2 — Entrenamiento (para S2, 02/05)
- [ ] Notebook `02_baseline.ipynb`: transfer learning head only, ~5 epochs
- [ ] Notebook `03_finetuning.ipynb`: fine-tuning completo con early stopping
- [ ] Alcanzar Top-1 > 75% en test set
- [ ] Pipeline imagen→calorías funcionando end-to-end
- [ ] Notebook `04_embeddings.ipynb`: t-SNE + retrieval KNN

### Fase 3 — Refinamiento y Entrega (para Clase 12, 30/05)
- [ ] Optimizar hiperparámetros si hay tiempo (augmentation, lr, scheduler)
- [ ] Notebook `05_evaluation.ipynb`: análisis de errores y demo final
- [ ] (Opcional) Extensión elegida
- [ ] Informe escrito (3 páginas): problema, datos, arquitectura, resultados
- [ ] Presentación final (12 minutos)

---

## 11. Papers de Referencia Clave

| Paper | Relevancia |
|-------|-----------|
| Bossard et al. (2014) — Food-101, ECCV | Dataset principal, benchmark de comparación. PDF: https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/static/bossard_eccv14_food-101.pdf |
| Meyers et al. (2015) — Im2Calories, ICCV (Google) | Paper fundacional del pipeline imagen→calorías. PDF: https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Meyers_Im2Calories_Towards_an_ICCV_2015_paper.pdf |
| Thames et al. (2021) — Nutrition5k, CVPR | Dataset con calorías reales por ingrediente. arXiv: https://arxiv.org/abs/2103.03375 |
| NutritionVerse (2024) | ViT multi-task para predicción simultánea de nutrientes. arXiv: https://arxiv.org/abs/2405.07814 |
| CNN+ViT Híbrido (2025), Nutrients MDPI | Arquitectura moderna con mapas de atención. https://www.mdpi.com/2072-6643/17/2/362 |

---

## 12. Datos Nutricionales — `data/nutrition_lookup.json`

Este es el contenido completo del archivo. Valores por 100g, fuente USDA FoodData Central. Crear este archivo antes de correr cualquier notebook de entrenamiento o inferencia.

```json
{
  "apple_pie":               {"calories": 237, "protein_g": 2.1,  "carbs_g": 34.0, "fat_g": 10.8, "fiber_g": 1.5, "portion_g": 125},
  "baby_back_ribs":          {"calories": 263, "protein_g": 22.0, "carbs_g": 5.0,  "fat_g": 17.0, "fiber_g": 0.0, "portion_g": 300},
  "baklava":                 {"calories": 428, "protein_g": 5.6,  "carbs_g": 52.0, "fat_g": 23.0, "fiber_g": 2.0, "portion_g": 80},
  "beef_carpaccio":          {"calories": 174, "protein_g": 20.0, "carbs_g": 2.0,  "fat_g": 9.0,  "fiber_g": 0.0, "portion_g": 100},
  "beef_tartare":            {"calories": 196, "protein_g": 20.0, "carbs_g": 3.0,  "fat_g": 11.0, "fiber_g": 0.0, "portion_g": 150},
  "beet_salad":              {"calories": 72,  "protein_g": 2.2,  "carbs_g": 11.0, "fat_g": 2.5,  "fiber_g": 2.8, "portion_g": 150},
  "beignets":                {"calories": 340, "protein_g": 5.0,  "carbs_g": 44.0, "fat_g": 16.0, "fiber_g": 1.0, "portion_g": 100},
  "bibimbap":                {"calories": 120, "protein_g": 7.5,  "carbs_g": 18.0, "fat_g": 2.5,  "fiber_g": 2.0, "portion_g": 400},
  "bread_pudding":           {"calories": 217, "protein_g": 6.1,  "carbs_g": 35.0, "fat_g": 6.3,  "fiber_g": 0.8, "portion_g": 150},
  "breakfast_burrito":       {"calories": 217, "protein_g": 9.5,  "carbs_g": 22.0, "fat_g": 10.0, "fiber_g": 2.5, "portion_g": 250},
  "bruschetta":              {"calories": 170, "protein_g": 5.5,  "carbs_g": 28.0, "fat_g": 4.5,  "fiber_g": 2.0, "portion_g": 100},
  "caesar_salad":            {"calories": 158, "protein_g": 5.8,  "carbs_g": 8.5,  "fat_g": 12.0, "fiber_g": 1.5, "portion_g": 200},
  "cannoli":                 {"calories": 365, "protein_g": 7.2,  "carbs_g": 40.0, "fat_g": 19.0, "fiber_g": 0.5, "portion_g": 100},
  "caprese_salad":           {"calories": 166, "protein_g": 10.0, "carbs_g": 5.0,  "fat_g": 12.0, "fiber_g": 0.8, "portion_g": 200},
  "carrot_cake":             {"calories": 370, "protein_g": 4.5,  "carbs_g": 52.0, "fat_g": 17.0, "fiber_g": 1.5, "portion_g": 100},
  "ceviche":                 {"calories": 103, "protein_g": 17.0, "carbs_g": 5.0,  "fat_g": 2.0,  "fiber_g": 1.0, "portion_g": 200},
  "cheesecake":              {"calories": 321, "protein_g": 6.0,  "carbs_g": 36.0, "fat_g": 17.0, "fiber_g": 0.4, "portion_g": 120},
  "cheese_plate":            {"calories": 371, "protein_g": 23.0, "carbs_g": 1.3,  "fat_g": 30.0, "fiber_g": 0.0, "portion_g": 100},
  "chicken_curry":           {"calories": 150, "protein_g": 16.0, "carbs_g": 8.0,  "fat_g": 6.0,  "fiber_g": 2.0, "portion_g": 300},
  "chicken_quesadilla":      {"calories": 270, "protein_g": 17.0, "carbs_g": 25.0, "fat_g": 11.0, "fiber_g": 1.5, "portion_g": 200},
  "chicken_wings":           {"calories": 290, "protein_g": 27.0, "carbs_g": 3.0,  "fat_g": 19.0, "fiber_g": 0.1, "portion_g": 200},
  "chocolate_cake":          {"calories": 371, "protein_g": 5.0,  "carbs_g": 53.0, "fat_g": 16.0, "fiber_g": 2.5, "portion_g": 100},
  "chocolate_mousse":        {"calories": 246, "protein_g": 4.5,  "carbs_g": 27.0, "fat_g": 14.0, "fiber_g": 1.5, "portion_g": 150},
  "churros":                 {"calories": 362, "protein_g": 5.0,  "carbs_g": 50.0, "fat_g": 16.0, "fiber_g": 2.0, "portion_g": 100},
  "clam_chowder":            {"calories": 100, "protein_g": 6.5,  "carbs_g": 11.0, "fat_g": 3.5,  "fiber_g": 0.5, "portion_g": 300},
  "club_sandwich":           {"calories": 294, "protein_g": 18.0, "carbs_g": 26.0, "fat_g": 12.0, "fiber_g": 2.0, "portion_g": 250},
  "crab_cakes":              {"calories": 179, "protein_g": 15.0, "carbs_g": 10.0, "fat_g": 8.5,  "fiber_g": 0.5, "portion_g": 150},
  "creme_brulee":            {"calories": 280, "protein_g": 4.5,  "carbs_g": 26.0, "fat_g": 18.0, "fiber_g": 0.0, "portion_g": 150},
  "croque_madame":           {"calories": 290, "protein_g": 16.0, "carbs_g": 20.0, "fat_g": 16.0, "fiber_g": 1.0, "portion_g": 250},
  "cup_cakes":               {"calories": 385, "protein_g": 4.5,  "carbs_g": 56.0, "fat_g": 17.0, "fiber_g": 0.8, "portion_g": 80},
  "deviled_eggs":            {"calories": 153, "protein_g": 8.5,  "carbs_g": 2.5,  "fat_g": 12.0, "fiber_g": 0.0, "portion_g": 120},
  "donuts":                  {"calories": 390, "protein_g": 5.0,  "carbs_g": 48.0, "fat_g": 20.0, "fiber_g": 1.5, "portion_g": 75},
  "dumplings":               {"calories": 180, "protein_g": 8.0,  "carbs_g": 26.0, "fat_g": 5.0,  "fiber_g": 1.5, "portion_g": 200},
  "edamame":                 {"calories": 121, "protein_g": 11.0, "carbs_g": 10.0, "fat_g": 5.2,  "fiber_g": 5.2, "portion_g": 150},
  "eggs_benedict":           {"calories": 254, "protein_g": 13.0, "carbs_g": 18.0, "fat_g": 14.0, "fiber_g": 1.0, "portion_g": 250},
  "escargots":               {"calories": 118, "protein_g": 16.0, "carbs_g": 2.0,  "fat_g": 5.5,  "fiber_g": 0.0, "portion_g": 150},
  "falafel":                 {"calories": 333, "protein_g": 13.0, "carbs_g": 31.0, "fat_g": 18.0, "fiber_g": 5.0, "portion_g": 150},
  "filet_mignon":            {"calories": 287, "protein_g": 26.0, "carbs_g": 0.0,  "fat_g": 19.0, "fiber_g": 0.0, "portion_g": 200},
  "fish_and_chips":          {"calories": 290, "protein_g": 14.0, "carbs_g": 32.0, "fat_g": 11.0, "fiber_g": 2.5, "portion_g": 350},
  "foie_gras":               {"calories": 462, "protein_g": 11.0, "carbs_g": 4.7,  "fat_g": 43.0, "fiber_g": 0.0, "portion_g": 80},
  "french_fries":            {"calories": 312, "protein_g": 3.4,  "carbs_g": 41.0, "fat_g": 15.0, "fiber_g": 3.8, "portion_g": 150},
  "french_onion_soup":       {"calories": 59,  "protein_g": 2.5,  "carbs_g": 8.0,  "fat_g": 2.0,  "fiber_g": 0.8, "portion_g": 300},
  "french_toast":            {"calories": 229, "protein_g": 8.2,  "carbs_g": 27.0, "fat_g": 10.0, "fiber_g": 1.2, "portion_g": 150},
  "fried_calamari":          {"calories": 175, "protein_g": 15.0, "carbs_g": 8.5,  "fat_g": 8.5,  "fiber_g": 0.3, "portion_g": 150},
  "fried_rice":              {"calories": 163, "protein_g": 4.8,  "carbs_g": 27.0, "fat_g": 4.1,  "fiber_g": 1.0, "portion_g": 300},
  "frozen_yogurt":           {"calories": 159, "protein_g": 4.0,  "carbs_g": 31.0, "fat_g": 2.3,  "fiber_g": 0.0, "portion_g": 150},
  "garlic_bread":            {"calories": 350, "protein_g": 7.5,  "carbs_g": 43.0, "fat_g": 16.0, "fiber_g": 2.0, "portion_g": 80},
  "gnocchi":                 {"calories": 130, "protein_g": 3.5,  "carbs_g": 27.0, "fat_g": 0.8,  "fiber_g": 1.0, "portion_g": 300},
  "greek_salad":             {"calories": 74,  "protein_g": 2.7,  "carbs_g": 7.5,  "fat_g": 4.0,  "fiber_g": 1.5, "portion_g": 250},
  "grilled_cheese_sandwich": {"calories": 350, "protein_g": 14.0, "carbs_g": 32.0, "fat_g": 19.0, "fiber_g": 2.0, "portion_g": 200},
  "grilled_salmon":          {"calories": 208, "protein_g": 28.0, "carbs_g": 0.0,  "fat_g": 10.0, "fiber_g": 0.0, "portion_g": 200},
  "guacamole":               {"calories": 160, "protein_g": 2.0,  "carbs_g": 9.0,  "fat_g": 15.0, "fiber_g": 6.7, "portion_g": 100},
  "gyoza":                   {"calories": 185, "protein_g": 8.5,  "carbs_g": 23.0, "fat_g": 6.5,  "fiber_g": 1.5, "portion_g": 200},
  "hamburger":               {"calories": 295, "protein_g": 17.0, "carbs_g": 24.0, "fat_g": 14.0, "fiber_g": 1.2, "portion_g": 250},
  "hot_and_sour_soup":       {"calories": 49,  "protein_g": 4.5,  "carbs_g": 5.0,  "fat_g": 1.5,  "fiber_g": 0.5, "portion_g": 300},
  "hot_dog":                 {"calories": 290, "protein_g": 11.0, "carbs_g": 23.0, "fat_g": 17.0, "fiber_g": 1.0, "portion_g": 150},
  "huevos_rancheros":        {"calories": 165, "protein_g": 9.5,  "carbs_g": 16.0, "fat_g": 7.5,  "fiber_g": 3.5, "portion_g": 300},
  "hummus":                  {"calories": 166, "protein_g": 8.0,  "carbs_g": 14.0, "fat_g": 9.6,  "fiber_g": 6.0, "portion_g": 100},
  "ice_cream":               {"calories": 207, "protein_g": 3.5,  "carbs_g": 24.0, "fat_g": 11.0, "fiber_g": 0.0, "portion_g": 150},
  "lasagna":                 {"calories": 166, "protein_g": 9.8,  "carbs_g": 17.0, "fat_g": 6.3,  "fiber_g": 2.0, "portion_g": 350},
  "lobster_bisque":          {"calories": 102, "protein_g": 6.5,  "carbs_g": 9.0,  "fat_g": 4.5,  "fiber_g": 0.3, "portion_g": 300},
  "lobster_roll_sandwich":   {"calories": 280, "protein_g": 18.0, "carbs_g": 28.0, "fat_g": 10.0, "fiber_g": 1.5, "portion_g": 250},
  "macaroni_and_cheese":     {"calories": 358, "protein_g": 14.0, "carbs_g": 44.0, "fat_g": 14.0, "fiber_g": 1.5, "portion_g": 300},
  "macarons":                {"calories": 426, "protein_g": 5.5,  "carbs_g": 64.0, "fat_g": 17.0, "fiber_g": 1.0, "portion_g": 50},
  "miso_soup":               {"calories": 40,  "protein_g": 3.0,  "carbs_g": 5.0,  "fat_g": 1.2,  "fiber_g": 0.5, "portion_g": 250},
  "mussels":                 {"calories": 172, "protein_g": 24.0, "carbs_g": 7.4,  "fat_g": 4.5,  "fiber_g": 0.0, "portion_g": 250},
  "nachos":                  {"calories": 346, "protein_g": 8.0,  "carbs_g": 40.0, "fat_g": 17.0, "fiber_g": 3.5, "portion_g": 200},
  "omelette":                {"calories": 154, "protein_g": 11.0, "carbs_g": 1.5,  "fat_g": 12.0, "fiber_g": 0.0, "portion_g": 200},
  "onion_rings":             {"calories": 320, "protein_g": 4.5,  "carbs_g": 38.0, "fat_g": 16.0, "fiber_g": 2.0, "portion_g": 150},
  "oysters":                 {"calories": 81,  "protein_g": 9.5,  "carbs_g": 4.7,  "fat_g": 2.5,  "fiber_g": 0.0, "portion_g": 150},
  "pad_thai":                {"calories": 194, "protein_g": 9.0,  "carbs_g": 30.0, "fat_g": 4.5,  "fiber_g": 2.0, "portion_g": 350},
  "paella":                  {"calories": 148, "protein_g": 10.0, "carbs_g": 19.0, "fat_g": 3.5,  "fiber_g": 1.0, "portion_g": 400},
  "pancakes":                {"calories": 227, "protein_g": 6.4,  "carbs_g": 38.0, "fat_g": 5.8,  "fiber_g": 1.5, "portion_g": 200},
  "panna_cotta":             {"calories": 220, "protein_g": 3.5,  "carbs_g": 22.0, "fat_g": 14.0, "fiber_g": 0.0, "portion_g": 150},
  "peking_duck":             {"calories": 337, "protein_g": 19.0, "carbs_g": 0.0,  "fat_g": 28.0, "fiber_g": 0.0, "portion_g": 200},
  "pho":                     {"calories": 65,  "protein_g": 6.0,  "carbs_g": 8.0,  "fat_g": 1.5,  "fiber_g": 0.5, "portion_g": 500},
  "pizza":                   {"calories": 266, "protein_g": 11.0, "carbs_g": 33.0, "fat_g": 10.0, "fiber_g": 2.3, "portion_g": 250},
  "pork_chop":               {"calories": 231, "protein_g": 28.0, "carbs_g": 0.0,  "fat_g": 12.0, "fiber_g": 0.0, "portion_g": 250},
  "poutine":                 {"calories": 240, "protein_g": 8.0,  "carbs_g": 30.0, "fat_g": 10.0, "fiber_g": 2.0, "portion_g": 400},
  "prime_rib":               {"calories": 340, "protein_g": 26.0, "carbs_g": 0.0,  "fat_g": 26.0, "fiber_g": 0.0, "portion_g": 300},
  "pulled_pork_sandwich":    {"calories": 254, "protein_g": 18.0, "carbs_g": 26.0, "fat_g": 8.0,  "fiber_g": 2.0, "portion_g": 250},
  "ramen":                   {"calories": 436, "protein_g": 21.0, "carbs_g": 57.0, "fat_g": 14.0, "fiber_g": 3.0, "portion_g": 500},
  "ravioli":                 {"calories": 166, "protein_g": 8.0,  "carbs_g": 26.0, "fat_g": 3.8,  "fiber_g": 2.0, "portion_g": 300},
  "red_velvet_cake":         {"calories": 367, "protein_g": 4.5,  "carbs_g": 54.0, "fat_g": 15.0, "fiber_g": 1.0, "portion_g": 100},
  "risotto":                 {"calories": 166, "protein_g": 5.0,  "carbs_g": 26.0, "fat_g": 5.0,  "fiber_g": 1.0, "portion_g": 350},
  "samosa":                  {"calories": 308, "protein_g": 6.5,  "carbs_g": 34.0, "fat_g": 16.0, "fiber_g": 3.0, "portion_g": 150},
  "sashimi":                 {"calories": 127, "protein_g": 21.0, "carbs_g": 0.0,  "fat_g": 4.5,  "fiber_g": 0.0, "portion_g": 150},
  "scallops":                {"calories": 111, "protein_g": 20.0, "carbs_g": 5.4,  "fat_g": 0.9,  "fiber_g": 0.0, "portion_g": 200},
  "seaweed_salad":           {"calories": 45,  "protein_g": 1.5,  "carbs_g": 8.0,  "fat_g": 0.8,  "fiber_g": 0.7, "portion_g": 150},
  "shrimp_and_grits":        {"calories": 225, "protein_g": 14.0, "carbs_g": 25.0, "fat_g": 8.0,  "fiber_g": 1.5, "portion_g": 350},
  "spaghetti_bolognese":     {"calories": 229, "protein_g": 14.0, "carbs_g": 28.0, "fat_g": 7.0,  "fiber_g": 2.5, "portion_g": 400},
  "spaghetti_carbonara":     {"calories": 326, "protein_g": 16.0, "carbs_g": 38.0, "fat_g": 12.0, "fiber_g": 2.0, "portion_g": 350},
  "spring_rolls":            {"calories": 155, "protein_g": 5.5,  "carbs_g": 22.0, "fat_g": 5.0,  "fiber_g": 1.5, "portion_g": 150},
  "steak":                   {"calories": 271, "protein_g": 26.0, "carbs_g": 0.0,  "fat_g": 18.0, "fiber_g": 0.0, "portion_g": 250},
  "strawberry_shortcake":    {"calories": 297, "protein_g": 4.0,  "carbs_g": 45.0, "fat_g": 12.0, "fiber_g": 1.5, "portion_g": 150},
  "sushi":                   {"calories": 143, "protein_g": 6.0,  "carbs_g": 24.0, "fat_g": 3.0,  "fiber_g": 0.8, "portion_g": 200},
  "tacos":                   {"calories": 226, "protein_g": 12.0, "carbs_g": 21.0, "fat_g": 10.0, "fiber_g": 2.5, "portion_g": 200},
  "takoyaki":                {"calories": 205, "protein_g": 9.5,  "carbs_g": 24.0, "fat_g": 8.0,  "fiber_g": 0.5, "portion_g": 200},
  "tiramisu":                {"calories": 283, "protein_g": 5.8,  "carbs_g": 27.0, "fat_g": 17.0, "fiber_g": 0.3, "portion_g": 150},
  "tuna_tartare":            {"calories": 144, "protein_g": 20.0, "carbs_g": 3.5,  "fat_g": 5.5,  "fiber_g": 0.5, "portion_g": 150},
  "waffles":                 {"calories": 291, "protein_g": 7.9,  "carbs_g": 37.0, "fat_g": 13.0, "fiber_g": 1.3, "portion_g": 200}
}
```
