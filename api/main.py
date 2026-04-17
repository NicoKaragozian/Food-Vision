"""
Esqueleto de API FastAPI para Food Vision.
No forma parte del TP — dejar como referencia para uso futuro.

Para correr (cuando se quiera):
    pip install fastapi uvicorn python-multipart
    cd TP\ Final
    uvicorn api.main:app --reload
"""

# from fastapi import FastAPI, UploadFile, File
# from fastapi.responses import JSONResponse
# from PIL import Image
# import io
# import sys
# sys.path.append("..")
#
# from src.pipeline import FoodVisionPipeline
#
# app = FastAPI(title="Food Vision API", version="1.0")
# pipe = FoodVisionPipeline()  # carga el mejor checkpoint disponible
#
#
# @app.get("/")
# def root():
#     return {"status": "ok", "model": pipe.backbone}
#
#
# @app.post("/predict")
# async def predict(file: UploadFile = File(...), portion_g: float = None):
#     """
#     Recibe una imagen y devuelve la categoría + estimación nutricional.
#
#     Ejemplo con curl:
#         curl -X POST "http://localhost:8000/predict" \
#              -F "file=@foto_pizza.jpg"
#     """
#     contents = await file.read()
#     image = Image.open(io.BytesIO(contents)).convert("RGB")
#     result = pipe.analyze(image, portion_g=portion_g)
#     return JSONResponse(content=result)
