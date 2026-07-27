"""Comprobación rápida de que cada modelo produce embeddings útiles.

Uso:  python validate_models.py                # ambos modelos
      python validate_models.py BiomedCLIP     # solo uno

Toma 50 pares imagen–leyenda al azar (semilla fija) y mide si cada imagen
recupera su propia leyenda entre las 50. Con un conjunto tan pequeño la tarea es
fácil: un modelo sano supera holgadamente el 30 % de Recall@1, mientras que el
azar daría 2 %.

Se reporta además `dispersion`: la similitud coseno media entre imágenes
distintas. Un valor cercano a 1 delata un codificador visual degenerado —mapea
todo al mismo punto— y fue precisamente el síntoma que llevó a descartar MedCLIP
de este trabajo. Ejecutar esta comprobación antes de lanzar el pipeline completo
evita gastar horas de GPU en embeddings inservibles.
"""
import os
import sys

import pandas as pd
import torch
from PIL import Image

import models_lib as ML

N = 50
UMBRAL_R1 = 0.30
DATA = "MultiCaReDataset"


def cargar_pares():
    cl = pd.read_csv(f"{DATA}/captions_and_labels.csv",
                     usecols=["file", "caption"]).dropna(subset=["caption"])
    muestra = cl.sample(N, random_state=1).reset_index(drop=True)
    rutas = [os.path.join(DATA, f[:4], f[:5], f) for f in muestra.file]
    return [Image.open(p).convert("RGB") for p in rutas], list(muestra.caption)


def comprobar(nombre, imagenes, leyendas):
    enc_img, enc_txt = ML.load_encoders(nombre)
    IE, TE = enc_img(imagenes), enc_txt(leyendas)

    similitud_imgs = IE @ IE.T
    dispersion = similitud_imgs[~torch.eye(len(IE), dtype=bool)].mean().item()

    S = IE @ TE.T
    orden = S.argsort(1, descending=True)
    posicion = (orden == torch.arange(len(IE))[:, None]).float().argmax(1)
    r1 = (posicion == 0).float().mean().item()
    r5 = (posicion < 5).float().mean().item()

    estado = "OK   " if r1 >= UMBRAL_R1 else "FALLO"
    print(f"[{estado}] {nombre:12s} R@1={r1:.2f}  R@5={r5:.2f}  "
          f"rango_mediano={int(posicion.median()) + 1}  dispersion={dispersion:.3f}")
    return r1 >= UMBRAL_R1


if __name__ == "__main__":
    modelos = sys.argv[1:] or list(ML.MODELOS)
    desconocidos = [m for m in modelos if m not in ML.MODELOS]
    if desconocidos:
        raise SystemExit(f"Modelo no soportado: {desconocidos}. Opciones: {list(ML.MODELOS)}")

    imagenes, leyendas = cargar_pares()
    print(f"{N} pares imagen–leyenda, dispositivo: {ML.DEVICE}\n")
    ok = all([comprobar(m, imagenes, leyendas) for m in modelos])
    raise SystemExit(0 if ok else 1)
