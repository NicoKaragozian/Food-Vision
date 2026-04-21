# Food Vision — Roadmap de Mejoras
**MIA304 · Visión y Percepción Computarizada · UdeSA 2026**  
*Documento de planificación técnica. El código se implementa por separado siguiendo este plan.*

---

## Estado Actual (Entrega Intermedia)

El sistema actual resuelve la mitad del problema:

- **Clasificación:** EfficientNet-B0 fine-tuneado sobre Food-101. Top-1: 82.86%, Top-5: 95.83%.
- **Nutrición:** lookup fijo en `nutrition_lookup.json` (USDA). Porción siempre fija por categoría.
- **Limitación central:** el modelo ve la imagen entera y produce *una sola clase*. No sabe dónde está la comida, cuánta hay, ni si hay varios ingredientes distintos.
- **Error calórico:** MAPE 10.6% — buen número, pero construido sobre una porción inventada.

El pipeline completo actual es: imagen → EfficientNet-B0 → clase → lookup → calorías fijas.

---

## Qué Queremos Construir

Un sistema que dado una foto de un plato de comida pueda:

1. **Detectar** cada alimento individualmente dentro del plato (no solo clasificar la imagen completa).
2. **Segmentar** con precisión pixel-wise qué área ocupa cada ingrediente.
3. **Estimar la porción** a partir del área real del ingrediente y una referencia de escala (el plato).
4. **Clasificar** cada ingrediente con el clasificador ya entrenado o uno mejor.
5. **Reportar métricas honestas** que reflejen tanto la calidad de la detección como la del error nutricional.

La narrativa del informe final es: partimos de un clasificador de imagen completa con porción fija, y llegamos a un detector multi-ingrediente con estimación de gramos basada en geometría visual.

---

## Cronograma de Implementación

Cada mejora está anclada a una clase del curso que provee el marco teórico.

| Paso | Clase | Fecha | Tema del curso |
|------|-------|-------|----------------|
| 1 — Métricas completas | 6 ✅ | 18/04 | Detección, Precision-Recall, mAP |
| 2 — Detección multi-alimento | 6 ✅ | 18/04 | YOLOv8, bounding boxes, IoU |
| 3 — Segmentación con SAM | 7 | 25/04 | Segmentación semántica |
| 4 — Estimación de porción por área | 7 | 25/04 | Segmentación + geometría |
| 5 — Backbone transformer (CLIP/ViT) | 9 | 09/05 | Transformers en visión |
| 6 — Detección open-set (Grounding DINO) | 9 | 09/05 | Transformers en visión |
| Seguimiento S2 | 8 | 02/05 | Mostrar pasos 1-3 funcionando |
| Seguimiento S3 | 10 | 16/05 | Sistema completo cerrado |
| Entrega final | 12 | 30/05 | Informe + presentación |

---

## Paso 1 — Métricas de Evaluación Completas

### Por qué es el primer paso

La entrega intermedia reporta Top-1 y Top-5 accuracy, y MAPE calórico. Son métricas necesarias pero insuficientes. Antes de mejorar el modelo, hay que tener un sistema de medición honesto que permita comparar cada versión futura con la anterior. Sin esto, no sabemos si una mejora realmente mejora.

### Qué métricas agregar al notebook 05

**Precision y Recall por clase**

La accuracy global del 82.86% oculta diferencias enormes entre clases. El notebook 05 ya identificó que `steak` tiene 54.8% de accuracy, pero no sabemos su Precision ni su Recall. La diferencia importa:

- **Precision baja:** el modelo predice "steak" cuando no lo es (muchos falsos positivos de steak → sobreestima calorías de carne)
- **Recall bajo:** el modelo no predice "steak" cuando sí lo es (muchos falsos negativos → subestima calorías de carne)

Estos dos errores tienen consecuencias nutricionales distintas y el informe debería distinguirlos.

**F1-score por clase**

Media armónica de Precision y Recall. Útil para resumir el desempeño de una clase en un solo número cuando hay que comparar. Especialmente relevante para las 20 clases con menor accuracy del notebook 05 — algunas tendrán Precision baja, otras Recall bajo, el F1 las nivela para comparar.

**Curva Precision-Recall del clasificador**

El notebook 05 ya construye `all_probs` y `all_preds` sobre el test set completo. Con esos datos se puede trazar la curva PR variando el umbral de confianza. La curva PR del clasificador es el equivalente directo a la curva PR del detector vista en Clase 6 — la misma idea aplicada a clasificación multi-clase.

El punto de operación óptimo (umbral = 0.39 para 90% accuracy con 85.9% de cobertura, ya calculado en notebook 05) debería mostrarse explícitamente en esta curva.

**Matriz de confusión mejorada**

Ya existe en el notebook 05 pero puede enriquecerse mostrando los errores más costosos en términos calóricos: confundir `chocolate_cake` con `red_velvet_cake` tiene bajo impacto calórico; confundir `foie_gras` con `edamame` tiene impacto enorme.

### Lo que esto habilita

Una vez que tenemos Precision, Recall, F1 y curva PR bien calculados para el clasificador actual, tenemos la línea de base completa contra la que comparar todas las mejoras futuras. Cada paso siguiente debería poder reportar estas mismas métricas para el componente que agrega.

---

## Paso 2 — Detección Multi-Alimento con YOLOv8

### El problema que resuelve

Un plato de bife con papas y huevo frito produce hoy una sola predicción: "baby_back_ribs" con 38.8% de confianza. El sistema ignora las papas y el huevo, que suman una cantidad significativa de calorías. Además, la confianza del 38.8% muestra que el modelo está inseguro — tiene sentido, porque la imagen contiene tres cosas distintas.

La detección de objetos resuelve esto: en lugar de procesar la imagen completa, localiza cada ingrediente con un bounding box y los clasifica por separado.

### Arquitectura propuesta: detector + clasificador en cascada

El enfoque más directo (y que no requiere reentrenar nada) es una arquitectura de dos etapas:

**Etapa 1 — YOLOv8 como detector de regiones**

YOLOv8 preentrenado en COCO puede detectar regiones genéricas. Aunque COCO no tiene categorías de comida específicas, el modelo detecta bien "objetos en un plato" como regiones diferenciadas. La idea es usar YOLOv8n (la versión más liviana, suficiente para este propósito) para obtener bounding boxes de las distintas regiones del plato.

Alternativa más robusta: existe un subconjunto de datasets con comida anotada con bounding boxes (UECFOOD-256, FoodDet-10k) que permitiría fine-tunear YOLOv8 directamente sobre categorías de comida. Esto mejoraría la detección de ingredientes específicos pero requiere tiempo de anotación y entrenamiento.

**Etapa 2 — EfficientNet-B0 sobre cada crop**

Una vez que YOLOv8 produce bounding boxes, se recorta cada región de la imagen original y se pasa por el clasificador EfficientNet-B0 ya entrenado. El clasificador ya sabe distinguir 101 categorías de comida — simplemente ahora opera sobre regiones más pequeñas y específicas en lugar de la imagen entera.

El resultado: cada región del plato tiene su propia predicción de clase y confianza. Las calorías se suman.

### Por qué este enfoque y no otros

**¿Por qué no reentrenar EfficientNet-B0 como detector?** EfficientNet es un clasificador, no un detector. Convertirlo en detector requeriría agregar una cabeza de detección (YOLO-style o Faster R-CNN-style), anotar bounding boxes en Food-101 (que no existen), y reentrenar. Mucho trabajo para este punto del proyecto.

**¿Por qué no usar un detector fine-tuneado sobre comida desde el inicio?** Requiere anotaciones de bounding boxes. Food-101 solo tiene etiquetas de clase, no coordenadas de objetos. La cascada YOLOv8 + EfficientNet-B0 evita este problema.

### Métricas a reportar para este paso

- **Precision, Recall, F1 del detector** a IoU threshold 0.50 (PASCAL VOC)
- **mAP@50** del detector sobre un subset de imágenes con múltiples ingredientes
- **Comparativa calórica:** imagen completa vs. suma de ingredientes detectados
- **Análisis de FP y FN:** ¿qué tipos de ingredientes se pierden más? ¿cuáles generan más detecciones falsas?

### Relación con Clase 6

Todo el marco conceptual (IoU para definir TP/FP, curva PR, mAP) viene directamente de la Clase 6 y Práctica 6. La diferencia es que ahora lo aplicamos a detección de comida real en lugar del dataset COCO genérico.

---

## Paso 3 — Segmentación con SAM

### El problema que resuelve

Un bounding box es un rectángulo que envuelve el ingrediente, pero la comida rara vez tiene forma rectangular. Un trozo de carne irregular puede ocupar solo el 60% del área del bounding box. Si estimamos la porción a partir del área del rectángulo, sobreestimamos los gramos.

La segmentación produce una máscara pixel-wise: sabe exactamente qué píxeles pertenecen al ingrediente. Esto permite calcular el área real con mucha mayor precisión.

### Segment Anything Model (SAM) — Meta AI, 2023

SAM es un modelo de segmentación fundacional entrenado sobre más de 1 billón de máscaras. Su característica clave es ser **promptable**: acepta como entrada un punto, un bounding box, o texto, y devuelve una máscara de segmentación precisa de la región indicada.

Para Food Vision, el pipeline natural es usar los bounding boxes del paso anterior como prompts de SAM. El flujo sería:

```
Imagen → YOLOv8 → bounding box del steak
                → SAM recibe el box como prompt
                → SAM devuelve máscara exacta del steak (solo los píxeles de carne)
                → área real en cm² si conocemos la escala
```

SAM no clasifica qué es la comida (eso ya lo hace EfficientNet-B0), solo delinea con precisión dónde están los bordes. Esta división de responsabilidades es limpia y evita reentrenar SAM.

### FoodSAM — la combinación validada por la literatura

El paper FoodSAM (Lan et al., 2023) validó exactamente este enfoque: SAM por sí solo falla al intentar clasificar ingredientes de comida (los confunde por textura y color), pero combinado con un clasificador semántico externo produce segmentaciones de alta calidad. El clasificador guía a SAM sobre qué regiones son relevantes.

IngredSAM (2024) va un paso más allá con one-shot segmentation: usando una imagen de referencia del ingrediente como prompt, segmenta ese ingrediente en imágenes nuevas. Supera al estado del arte en FoodSeg103 en 2.85% mIoU. Relevante si se quiere segmentación sin bounding boxes.

### Dataset de evaluación: FoodSeg103

FoodSeg103 (Wu et al., 2021) es el benchmark estándar para segmentación de comida: 9,490 imágenes, 103 categorías, con máscaras pixel-wise verificadas. Si se implementa segmentación, conviene evaluar sobre este dataset además de Food-101 para poder comparar con la literatura.

### Métricas a reportar para este paso

- **mIoU (mean Intersection over Union):** métrica estándar de segmentación semántica. Equivale al mAP en detección. Un mIoU de 0.70 significa que en promedio el 70% de los píxeles se asignan a la clase correcta.
- **Dice coefficient por clase:** similar al F1-score pero calculado sobre máscaras. Muy usado en papers de segmentación de comida, permite comparación directa con FoodSAM e IngredSAM.
- **Precisión del área estimada:** comparar el área en cm² calculada a partir de la máscara vs. el área real de la porción (si se tiene ground truth).

---

## Paso 4 — Estimación de Porción por Área

### El problema que resuelve

Aun con la máscara exacta del ingrediente, sabemos el área en píxeles pero no en gramos. Para pasar de píxeles a gramos necesitamos una referencia de escala en la imagen.

### Estrategia: el plato como referencia

Un plato de restaurant estándar tiene aproximadamente 28cm de diámetro. Si YOLOv8 detecta el plato (COCO tiene la clase "dining table" y "bowl" que ayuda), podemos calcular cuántos píxeles equivalen a 1 cm y escalar el área de cada ingrediente.

El cálculo tiene varias capas de incertidumbre que hay que ser honestos sobre ellas en el informe:

1. **Incertidumbre en el diámetro del plato:** no todos los platos son iguales. Usar 28cm como estándar introduce un error sistemático.
2. **Proyección 2D → área 3D:** la comida tiene altura, no es plana. Una pila de papas fritas puede tener el mismo área 2D que un solo trozo pero el triple de volumen.
3. **Densidad variable:** la densidad de la pizza es muy distinta a la de la ensalada. Para convertir volumen estimado a gramos se necesita la densidad por categoría.

La estimación de gramos por área es mejor que una porción fija, pero tiene error propio. El MAPE calórico del sistema mejorado debería compararse con el 10.6% actual para ver si realmente mejora.

### Nivel avanzado: profundidad monocular

Los sistemas del estado del arte (MFP3D, 2024; "From Pixels to Calories", Vinod & Zhu, 2025) usan un modelo de estimación de profundidad monocular sobre la misma imagen para reconstruir el volumen 3D de la comida. Modelos como DepthAnything v2 producen un mapa de profundidad desde una sola imagen sin hardware especial.

Con profundidad + segmentación se puede estimar el volumen real de cada ingrediente, multiplicar por su densidad, y obtener los gramos. Este es el pipeline completo del estado del arte y el objetivo a largo plazo del proyecto.

La complejidad de implementar depth monocular es alta. Para el TP puede mencionarse como trabajo futuro con referencia a los papers, mostrando que se entiende el estado del arte aunque no se implemente completo.

---

## Paso 5 — Backbone Transformer: CLIP o ViT

### Por qué los transformers superan a las CNNs en clasificación de comida

EfficientNet-B0 procesa la imagen localmente: cada filtro convolucional ve una ventana pequeña de píxeles. Para relacionar la textura de la salsa en un costado con el tipo de carne en el centro, necesita apilar muchas capas hasta que el campo receptivo sea suficientemente grande.

Un Vision Transformer (ViT) divide la imagen en parches de 16×16 píxeles y aplica self-attention entre todos los parches desde la primera capa. Cualquier parte del plato puede influir directamente en la representación de cualquier otra parte. Para comida, donde el contexto global (qué hay alrededor del ingrediente principal, qué tipo de plato es, cómo está presentado) es informativo, esto es una ventaja real.

### CLIP — el caso más directo

CLIP (OpenAI, 2021) preentrenó un ViT y un encoder de texto de forma contrastiva sobre 400 millones de pares imagen-texto de internet. El espacio de embedding resultante es tan rico que CLIP puede clasificar imágenes en categorías que nunca vio durante el entrenamiento, solo comparando el embedding de la imagen con el embedding del texto de cada clase.

Para Food-101, esto significa: en lugar de reentrenar una cabeza de 101 clases, se compara el embedding de la imagen con los embeddings de los 101 nombres de clase ("a photo of pizza", "a photo of sushi"...) y se devuelve el más cercano. Sin ningún fine-tuning adicional, CLIP reporta ~85% Top-1 en Food-101 en benchmarks recientes — comparable o superior a nuestro EfficientNet fine-tuneado.

Con fine-tuning sobre Food-101, CLIP supera ampliamente a EfficientNet-B0. La razón es que el preentrenamiento multimodal imagen-texto da representaciones mucho más ricas que el preentrenamiento en ImageNet solo.

### DINOv2 — features de segmentación sin arquitectura explícita

DINOv2 (Meta AI, 2023) es un ViT preentrenado con aprendizaje auto-supervisado (sin etiquetas) sobre 142 millones de imágenes curadas. Sus features son extraordinariamente ricas: los tokens locales (uno por parche de 16×16) capturan semántica suficiente para hacer segmentación densa sin arquitectura encoder-decoder explícita.

Para Food Vision, DINOv2 es relevante como backbone para el clasificador y también porque sus features locales pueden usarse para segmentación sin SAM en algunos casos. Si se usa DINOv2 + cabeza lineal el resultado es comparable a modelos mucho más pesados.

### Conexión con Clase 9

La Clase 9 (09/05) cubre transformers en visión. Después de esa clase se va a tener el marco teórico completo para entender ViT, self-attention sobre parches, y por qué el `[CLS]` token es el equivalente del `avgpool` de EfficientNet. Ese es el momento para experimentar con CLIP o ViT como reemplazo del backbone.

---

## Paso 6 — Detección Open-Set con Grounding DINO

### El problema de anotar bounding boxes

El paso 2 (YOLOv8 en cascada) evita el problema de la anotación usando YOLOv8 preentrenado en COCO para detectar regiones genéricas. Pero COCO no tiene categorías específicas de comida. El detector puede confundir regiones, especialmente en platos con muchos ingredientes mezclados.

Fine-tunear YOLOv8 sobre Food-101 requeriría anotar bounding boxes en miles de imágenes — trabajo que no se hizo y que escapa al alcance del TP.

### Grounding DINO — detección guiada por texto

Grounding DINO (ECCV 2024) es un detector open-set que acepta descripciones en texto como input. En lugar de estar limitado a las clases de entrenamiento, puede detectar cualquier objeto descrito con palabras.

Para Food Vision, esto resuelve el problema de anotación de forma elegante: en lugar de anotar "aquí hay steak" con un bounding box, se le dice al modelo "detecta el steak en esta imagen" y él lo localiza. No necesita haber visto "steak" durante el entrenamiento — generaliza desde su preentrenamiento en lenguaje y visión.

La arquitectura combina un transformer de visión (backbone DINO) con un encoder de texto (BERT), fusiona ambas modalidades via cross-attention, y produce bounding boxes anclados al texto de entrada.

Grounding DINO + EfficientNet-B0 para clasificación fina puede ser el pipeline de detección más práctico para este TP: no requiere anotar nada, usa el clasificador ya entrenado, y aprovecha el poder del lenguaje para guiar la detección.

---

## Plan de Métricas para el Informe Final

El informe de 3 páginas debe reportar métricas comparativas que muestren la evolución del sistema. La tabla propuesta:

| Versión del sistema | Top-1 | Precision | Recall | F1 | MAPE calórico |
|---|---|---|---|---|---|
| Entrega intermedia (baseline, imagen completa, porción fija) | 82.86% | — | — | — | 10.6% |
| + Detección multi-ingrediente (YOLOv8 + EfficientNet) | — | mAP@50 | Recall@50 | F1@50 | TBD |
| + Segmentación (SAM) + porción por área | — | mAP@50 | Recall@50 | F1@50 | TBD |
| + Backbone transformer (CLIP/ViT) | TBD | TBD | TBD | TBD | TBD |

Los "TBD" se completan a medida que se implementa cada paso. La narrativa es siempre: "cada mejora técnica se refleja en una o más métricas". Si el MAPE calórico baja de 10.6% a X%, es porque la estimación de porciones mejoró. Si el mAP@50 es alto pero el MAPE no mejora, significa que detectamos bien pero las porciones siguen siendo el problema.

---

## Papers de Referencia Clave

Ordenados por relevancia directa al plan:

| Paper | Relevancia |
|---|---|
| Bossard et al. (2014) — Food-101, ECCV | El dataset base de todo el sistema |
| Tan & Le (2019) — EfficientNet, ICML | El backbone del clasificador actual |
| Jocher et al. (2023) — YOLOv8, Ultralytics | El detector del paso 2 |
| Kirillov et al. (2023) — SAM, Meta AI · [arXiv:2304.02643](https://arxiv.org/abs/2304.02643) | La segmentación del paso 3 |
| Lan et al. (2023) — FoodSAM · [arXiv:2308.05938](https://arxiv.org/abs/2308.05938) | Validación de SAM para comida |
| Meyers et al. (2015) — Im2Calories, ICCV | Motivación del problema (estimación calórica desde imagen) |
| Radford et al. (2021) — CLIP, OpenAI | Backbone transformer del paso 5 |
| Dosovitskiy et al. (2021) — ViT, Google Brain | Arquitectura transformer base |
| Liu et al. (2023) — Grounding DINO, ECCV 2024 · [arXiv:2303.05499](https://arxiv.org/abs/2303.05499) | Detección open-set del paso 6 |
| Vinod & Zhu (2024) — Food Portion via 3D Scaling, CVPRW · [PDF](https://openaccess.thecvf.com/content/CVPR2024W/MTF/papers/Vinod_Food_Portion_Estimation_via_3D_Object_Scaling_CVPRW_2024_paper.pdf) | Estimación de porción 3D |
| Håkansson et al. (2025) — Swedish Plate Model + YOLOv8 · [JMIR](https://formative.jmir.org/2025/1/e70124) | Validación de YOLOv8 para estimación de proporciones |

---

## Notas para la Implementación

- **El orden importa:** cada paso depende del anterior. No tiene sentido implementar SAM sin tener YOLOv8 funcionando, porque SAM necesita los bounding boxes como prompts.
- **Comparar siempre contra el sistema anterior:** cada notebook nuevo debe reportar las métricas del sistema actual y del mejorado side-by-side.
- **No tirar lo que funciona:** EfficientNet-B0 fine-tuneado sigue siendo el clasificador. Los pasos 2-6 lo envuelven con más contexto, no lo reemplazan (hasta el paso 5).
- **La porción fija puede ser un buen fallback:** si la estimación de porción por área falla (imagen con ángulo raro, plato no detectado), conviene tener el sistema actual como fallback en lugar de no responder.
- **El informe tiene 3 páginas:** no hace falta implementar todo. Con pasos 1, 2 y parte del 3 ya hay una historia de mejora sólida para presentar el 30/05.
