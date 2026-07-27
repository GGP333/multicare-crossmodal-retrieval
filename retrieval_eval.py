"""Evaluación de recuperación cross-modal en las tres direcciones.

Uso:  python retrieval_eval.py               # ambos modelos
      python retrieval_eval.py BiomedCLIP    # solo uno

Cada fila i de artifacts/images_index.parquet enlaza una imagen, su leyenda y el
caso clínico al que pertenece. Esa correspondencia es la que define el "acierto":

  I→T   imagen_i          -> candidatas: las 130.791 leyendas
                             acierto = leyenda_i (rel 2); misma caso (rel 1)
  T→I   leyenda_i         -> candidatas: las 130.791 imágenes
                             acierto = imagen_i (rel 2); mismo caso (rel 1)
  TI→C  fusión(img, leyenda) -> candidatos: los casos con imagen
                             acierto = caso de i (rel 2); tópico compartido (rel 1)

Evaluar contra las leyendas y no contra la narrativa completa evita una fuga: la
narrativa del caso contiene el texto de sus propias leyendas, de modo que buscar
"la narrativa más parecida a esta imagen" sería en parte una búsqueda de texto
contra sí mismo.

En TI→C la representación del caso objetivo se recalcula excluyendo la imagen de
consulta (leave-one-out). Sin esa exclusión el caso correcto contendría la propia
consulta en su promedio visual y el resultado estaría inflado.

Métricas: Recall@1/5/10 y MRR sobre el acierto exacto; nDCG@10 con relevancia
graduada (2 = exacto, 1 = parcialmente relevante).

Salida: artifacts/metrics_{modelo}.json
"""
import json
import math
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import models_lib as ML

DEV = ML.DEVICE
Ks = [1, 5, 10]
NK = 10  # profundidad del nDCG

# ----------------------------------------------------------------------
# Índices (independientes del modelo: se cargan una sola vez)
# ----------------------------------------------------------------------
cases = pd.read_parquet("artifacts/cases_index.parquet")
imgs = pd.read_parquet("artifacts/images_index.parquet")
img_case = imgs.case_idx.to_numpy().copy()  # fila-imagen -> fila-caso (copia: pandas
                                            # devuelve una vista de solo lectura y
                                            # torch.from_numpy avisaría de ello)
img_type = imgs.image_type.to_numpy()   # modalidad, para el desglose estratificado
topics = [set(t) for t in cases.topics]
q = np.load("artifacts/q_it_img.npy")   # filas-imagen usadas como consulta

# Cuántas filas aporta cada caso: es el número máximo de resultados con rel = 1
# que podría haber en I→T y T→I, y por tanto entra en el nDCG ideal.
filas_por_caso = defaultdict(int)
for c in img_case:
    filas_por_caso[c] += 1

# Índice invertido de tópicos, para contar casos parcialmente relevantes en TI→C.
casos_por_topico = defaultdict(set)
for ci, ts in enumerate(topics):
    for t in ts:
        casos_por_topico[t].add(ci)


def n_casos_con_topico(ci):
    s = set()
    for t in topics[ci]:
        s |= casos_por_topico[t]
    s.discard(ci)
    return len(s)


# ----------------------------------------------------------------------
# Métricas
# ----------------------------------------------------------------------
def dcg(rels):
    return sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(rels))


def idcg(n_rel2, n_rel1, K=NK):
    """nDCG ideal: primero los aciertos exactos, después los parcialmente relevantes."""
    ideal = ([2] * n_rel2 + [1] * n_rel1)[:K]
    return sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(ideal)) or 1.0


def resumir(rangos, ndcgs, estratos):
    rangos = np.array(rangos, float)
    ndcgs = np.array(ndcgs)
    estratos = np.array(estratos)
    res = {f"R@{k}": float((rangos <= k).mean()) for k in Ks}
    res["MRR"] = float((1 / rangos).mean())
    res[f"nDCG@{NK}"] = float(ndcgs.mean())
    res["n"] = int(len(rangos))

    # Desglose por modalidad de imagen: la métrica agregada esconde diferencias
    # grandes entre, por ejemplo, patología y radiología.
    por_modalidad = {}
    for s in np.unique(estratos):
        m = estratos == s
        por_modalidad[str(s)] = {"R@1": float((rangos[m] <= 1).mean()),
                                 "R@10": float((rangos[m] <= 10).mean()),
                                 "MRR": float((1 / rangos[m]).mean()),
                                 f"nDCG@{NK}": float(ndcgs[m].mean()),
                                 "n": int(m.sum())}
    res["by_modality"] = por_modalidad
    return res


# ----------------------------------------------------------------------
# Direcciones de recuperación
# ----------------------------------------------------------------------
@torch.no_grad()
def evaluar_fila_a_fila(Q, candidatos):
    """I→T (Q = imágenes, candidatos = leyendas) o T→I (al revés).

    El rango se obtiene contando cuántos candidatos superan al correcto, en vez de
    ordenar los 130.791: es exacto y mucho más barato.
    """
    rangos, ndcgs, estratos = [], [], []
    qt = torch.from_numpy(q).to(DEV)
    for s in range(0, len(q), 256):
        qq = q[s:s + 256]
        sc = Q[qt[s:s + 256]] @ candidatos.T
        correcto = sc.gather(1, torch.from_numpy(qq).to(DEV)[:, None]).squeeze(1)
        rk = (sc > correcto[:, None]).sum(1) + 1
        top = sc.topk(NK, 1).indices.cpu().numpy()
        for j in range(len(qq)):
            qi = int(qq[j])
            c = int(img_case[qi])
            rangos.append(int(rk[j]))
            rels = [2 if int(t) == qi else (1 if int(img_case[int(t)]) == c else 0)
                    for t in top[j]]
            ndcgs.append(dcg(rels) / idcg(1, min(filas_por_caso[c] - 1, NK)))
            estratos.append(img_type[qi])
    return resumir(rangos, ndcgs, estratos)


def fusionar(visual, textual, modo, alpha=0.5):
    if modo == "concat":
        return torch.cat([visual, textual], dim=1)
    if modo == "wavg":  # alpha = 1 -> solo imagen; alpha = 0 -> solo texto
        return F.normalize(alpha * visual + (1 - alpha) * textual)
    return F.normalize(visual + textual)  # nsum


@torch.no_grad()
def evaluar_TI2C(EI, EC, ET, modo, alpha=0.5):
    d = EI.shape[1]
    # Representación visual de cada caso: media de los embeddings de sus imágenes.
    sumas = torch.zeros(len(cases), d, device=DEV)
    cuenta = torch.zeros(len(cases), 1, device=DEV)
    ic = torch.from_numpy(img_case).to(DEV)
    sumas.index_add_(0, ic, EI)
    cuenta.index_add_(0, ic, torch.ones(len(imgs), 1, device=DEV))
    tiene_imagen = cuenta.squeeze(1) > 0
    media = torch.zeros_like(sumas)
    media[tiene_imagen] = F.normalize(sumas[tiene_imagen] / cuenta[tiene_imagen])

    candidatos = F.normalize(fusionar(media, ET, modo, alpha))
    qt = torch.from_numpy(q).to(DEV)
    Qv, Qc = EI[qt], EC[qt]
    Q = F.normalize(fusionar(Qv, Qc, modo, alpha))

    rangos, ndcgs, estratos = [], [], []
    for s in range(0, len(q), 256):
        qq = q[s:s + 256]
        b = len(qq)
        sc = Q[s:s + b] @ candidatos.T
        sc[:, ~tiene_imagen] = -1e4          # casos sin imagen no son candidatos
        caso_correcto = torch.from_numpy(img_case[qq]).to(DEV)

        # Leave-one-out: recalcular el caso correcto sin la imagen de consulta.
        n_img = cuenta[caso_correcto].squeeze(1)
        media_loo = torch.where(
            (n_img > 1)[:, None],
            F.normalize((sumas[caso_correcto] - Qv[s:s + b]) / (n_img[:, None] - 1).clamp(min=1)),
            torch.zeros_like(Qv[s:s + b]))
        cand_loo = F.normalize(fusionar(media_loo, ET[caso_correcto], modo, alpha))
        sc.scatter_(1, caso_correcto[:, None], (Q[s:s + b] * cand_loo).sum(1)[:, None])

        correcto = sc.gather(1, caso_correcto[:, None]).squeeze(1)
        rk = (sc > correcto[:, None]).sum(1) + 1
        top = sc.topk(NK, 1).indices.cpu().numpy()
        for j in range(b):
            ci = int(caso_correcto[j])
            tops_q = topics[ci]
            rangos.append(int(rk[j]))
            rels = [2 if int(t) == ci else (1 if tops_q and (topics[int(t)] & tops_q) else 0)
                    for t in top[j]]
            n1 = min(n_casos_con_topico(ci), NK) if tops_q else 0
            ndcgs.append(dcg(rels) / idcg(1, n1))
            estratos.append(img_type[int(qq[j])])
    return resumir(rangos, ndcgs, estratos)


# ----------------------------------------------------------------------
def cargar(ruta):
    """fp16 en disco -> fp32 en el dispositivo, renormalizado tras la conversión."""
    return F.normalize(torch.from_numpy(np.load(ruta).copy()).float()).to(DEV)


def evaluar(modelo):
    EI = cargar(f"artifacts/emb_{modelo}_img.npy")
    EC = cargar(f"artifacts/emb_{modelo}_cap.npy")
    ET = cargar(f"artifacts/emb_{modelo}_txt.npy")
    print(f"[{modelo}] {DEV} | img{tuple(EI.shape)} cap{tuple(EC.shape)} txt{tuple(ET.shape)}",
          flush=True)

    out = {}
    print(f"[{modelo}] I->T", flush=True)
    out["I2T"] = evaluar_fila_a_fila(EI, EC)
    print(f"[{modelo}] T->I", flush=True)
    out["T2I"] = evaluar_fila_a_fila(EC, EI)
    print(f"[{modelo}] TI->C concat", flush=True)
    out["TI2C_concat"] = evaluar_TI2C(EI, EC, ET, "concat")
    print(f"[{modelo}] TI->C suma normalizada", flush=True)
    out["TI2C_nsum"] = evaluar_TI2C(EI, EC, ET, "nsum")

    # Barrido del peso de fusión: los extremos son las líneas base unimodales.
    barrido = {}
    for a in [0.0, 0.25, 0.5, 0.75, 1.0]:
        r = evaluar_TI2C(EI, EC, ET, "wavg", a)
        barrido[f"{a:.2f}"] = {k: r[k] for k in ["R@1", "R@5", "R@10", "MRR", "nDCG@10"]}
        print(f"[{modelo}] TI->C wavg alpha={a:.2f}  R@1={r['R@1']:.3f}", flush=True)
    out["TI2C_alpha_sweep"] = barrido

    json.dump(out, open(f"artifacts/metrics_{modelo}.json", "w"), indent=2)
    for k, v in out.items():
        if "R@1" in v:
            print(f"  {k}: R@1={v['R@1']:.3f} R@5={v['R@5']:.3f} R@10={v['R@10']:.3f} "
                  f"MRR={v['MRR']:.3f} nDCG@10={v['nDCG@10']:.3f} (n={v['n']})")
    print(f"=== artifacts/metrics_{modelo}.json ===\n")

    del EI, EC, ET
    if DEV == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    modelos = sys.argv[1:] or list(ML.MODELOS)
    desconocidos = [m for m in modelos if m not in ML.MODELOS]
    if desconocidos:
        raise SystemExit(f"Modelo no soportado: {desconocidos}. Opciones: {list(ML.MODELOS)}")
    for m in modelos:
        evaluar(m)
