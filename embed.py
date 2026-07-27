"""Calcula los tres bancos de embeddings que consume la evaluación.

Uso:  python embed.py                  # ambos modelos
      python embed.py BiomedCLIP       # solo uno

Salidas (float16, L2-normalizadas, alineadas fila a fila con los índices de
artifacts/ que genera data_prep.py):

  artifacts/emb_{modelo}_img.npy   130.791 x 512   una fila por imagen
  artifacts/emb_{modelo}_cap.npy   130.791 x 512   la leyenda de esa misma imagen
  artifacts/emb_{modelo}_txt.npy    93.816 x 512   la narrativa de cada caso clínico
"""
import contextlib
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import models_lib as ML

IMG_BATCH = 256
TXT_BATCH = 512
WORKERS = min(8, os.cpu_count() or 1)


def media_precision(device):
    """fp16 en GPU (2x más rápido, sin efecto medible en la similitud coseno);
    precisión completa en CPU, donde autocast no aporta."""
    if device == "cuda":
        return torch.autocast("cuda", dtype=torch.float16)
    return contextlib.nullcontext()


class ImagenesDS(Dataset):
    """Lee y preprocesa imágenes en los workers del DataLoader.

    Devuelve también el índice de la fila para poder escribir el resultado en su
    posición correcta: con num_workers > 0 los lotes no llegan necesariamente en orden.
    """

    def __init__(self, paths, preprocess):
        self.paths = paths
        self.preprocess = preprocess

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            im = Image.open(self.paths[i]).convert("RGB")
            return self.preprocess(im), i, True
        except Exception:
            # Una imagen ilegible no debe abortar un cómputo de varios minutos:
            # se sustituye por una imagen en negro y se contabiliza como fallo.
            return torch.zeros(3, 224, 224), i, False


@torch.no_grad()
def embed_imagenes(bb, paths):
    dl = DataLoader(ImagenesDS(paths, bb.preprocess), batch_size=IMG_BATCH,
                    num_workers=WORKERS, pin_memory=(bb.device == "cuda"))
    out = np.zeros((len(paths), ML.DIM), dtype=np.float16)
    fallos, hechas, t0 = 0, 0, time.time()
    for x, idx, ok in dl:
        x = x.to(bb.device, non_blocking=True)
        with media_precision(bb.device):
            e = F.normalize(bb.model.encode_image(x).float())
        out[idx.numpy()] = e.cpu().numpy().astype(np.float16)
        fallos += int((~ok).sum())
        hechas += len(idx)
        if hechas % (IMG_BATCH * 40) < IMG_BATCH:
            vel = hechas / (time.time() - t0)
            print(f"    imágenes {hechas}/{len(paths)}  {vel:.0f}/s  "
                  f"ETA {(len(paths) - hechas) / vel / 60:.1f} min", flush=True)
    if fallos:
        print(f"    AVISO: {fallos} imágenes ilegibles, sustituidas por una imagen en negro")
    return out


@torch.no_grad()
def embed_textos(bb, textos):
    out = np.zeros((len(textos), ML.DIM), dtype=np.float16)
    for s in range(0, len(textos), TXT_BATCH):
        lote = textos[s:s + TXT_BATCH]
        toks = bb.tokenize(lote).to(bb.device)
        with media_precision(bb.device):
            e = F.normalize(bb.model.encode_text(toks).float())
        out[s:s + len(lote)] = e.cpu().numpy().astype(np.float16)
    return out


def procesar(nombre, imgs, cases):
    print(f"=== {nombre} ({ML.DEVICE}) ===", flush=True)
    bb = ML.load(nombre)
    t0 = time.time()

    for etiqueta, sufijo, datos in [
        ("narrativas de caso", "txt", cases.case_text.tolist()),
        ("leyendas de imagen", "cap", imgs.caption.fillna("").tolist()),
    ]:
        print(f"  {etiqueta} ({len(datos):,})", flush=True)
        np.save(f"artifacts/emb_{nombre}_{sufijo}.npy", embed_textos(bb, datos))

    print(f"  imágenes ({len(imgs):,})", flush=True)
    np.save(f"artifacts/emb_{nombre}_img.npy", embed_imagenes(bb, list(imgs.path)))

    print(f"=== {nombre} listo en {(time.time() - t0) / 60:.1f} min ===\n", flush=True)
    del bb
    if ML.DEVICE == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    modelos = sys.argv[1:] or list(ML.MODELOS)
    desconocidos = [m for m in modelos if m not in ML.MODELOS]
    if desconocidos:
        raise SystemExit(f"Modelo no soportado: {desconocidos}. Opciones: {list(ML.MODELOS)}")

    imgs = pd.read_parquet("artifacts/images_index.parquet")
    cases = pd.read_parquet("artifacts/cases_index.parquet")
    for nombre in modelos:
        procesar(nombre, imgs, cases)
