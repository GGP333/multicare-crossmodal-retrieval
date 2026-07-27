"""Figuras que reproducen imágenes del corpus.

Uso:  python make_example_figs.py

  figs/fig_examples.pdf     galería de las 7 modalidades de imagen
  figs/fig_qualitative.pdf  ejemplo cualitativo de recuperación T→I (BiomedCLIP)

Solo se reproducen imágenes con licencia CC BY, la única del corpus que permite
redistribución sin restricciones adicionales. El resto del dataset (CC BY-NC,
CC BY-NC-SA) se usa para calcular métricas, pero no se muestra.

Corre en CPU: la búsqueda de un ejemplo ilustrativo recorre unas pocas miles de
consultas, no el conjunto de evaluación completo.
"""
import os, numpy as np, pandas as pd, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

OUT = "figs"
DATA = "MultiCaReDataset"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 10, "savefig.bbox": "tight", "figure.dpi": 150})

cl = pd.read_csv(f"{DATA}/captions_and_labels.csv",
                 usecols=["file", "license", "image_type", "image_subtype",
                          "caption", "file_size"])
cl["path"] = cl.file.map(lambda fn: os.path.join(DATA, fn[:4], fn[:5], fn))
ccby = cl[cl.license == "CC BY"].reset_index(drop=True)

def load_sq(path, size=320):
    im = Image.open(path).convert("RGB")
    im.thumbnail((size, size))
    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    canvas.paste(im, ((size - im.width) // 2, (size - im.height) // 2))
    return canvas

# ----------------------------------------------------------------------
# 1. Galería de modalidades
# ----------------------------------------------------------------------
ORDER = [("radiology", "Radiología", None),
         ("pathology", "Patología", "h&e"),
         ("medical_photograph", "Fotografía médica", "oral_photograph"),
         ("endoscopy", "Endoscopía", None),
         ("ophthalmic_imaging", "Imagen oftálmica", None),
         ("electrography", "Electrografía", None),
         ("chart", "Gráficos", None)]

# semilla determinista para elegir ejemplos "medianos" (no diminutos)
def pick(modality, subtype=None, k=0):
    sub = ccby[ccby.image_type == modality]
    if subtype is not None and (sub.image_subtype == subtype).any():
        sub = sub[sub.image_subtype == subtype]
    sub = sub[sub.file_size.between(40000, 400000)]
    sub = sub.sort_values("file").reset_index(drop=True)
    return sub.path.iloc[k % len(sub)]

fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.9))
axes = axes.ravel()
for ax, (mod, label, sub) in zip(axes, ORDER):
    ax.imshow(load_sq(pick(mod, sub, k=3)))
    ax.set_title(label, fontsize=10)
    ax.axis("off")
axes[-1].axis("off")  # octava celda vacía
fig.tight_layout()
fig.savefig(f"{OUT}/fig_examples.pdf")
plt.close(fig)
print("fig_examples.pdf OK")

# ----------------------------------------------------------------------
# 2. Ejemplo cualitativo de recuperación T->I (BiomedCLIP)
# ----------------------------------------------------------------------
imgs = pd.read_parquet("artifacts/images_index.parquet")
imgs = imgs.merge(cl[["file", "license"]], on="file", how="left")
EI = torch.nn.functional.normalize(
    torch.from_numpy(np.load("artifacts/emb_BiomedCLIP_img.npy").copy()).float())
EC = torch.nn.functional.normalize(
    torch.from_numpy(np.load("artifacts/emb_BiomedCLIP_cap.npy").copy()).float())

ccby_mask = (imgs.license == "CC BY").to_numpy()

def good_query(modality):
    """Busca una consulta-caption CC BY de la modalidad cuyo top-1 sea su imagen
    y cuyos top-3 sean todos CC BY (para poder mostrarlos)."""
    cand = imgs.index[(imgs.image_type == modality) & ccby_mask &
                      (imgs.caption.str.len() > 40)].to_numpy()
    for q in cand[:4000]:
        sc = EC[q] @ EI.T
        top = torch.topk(sc, 4).indices.numpy()
        if top[0] == q and all(ccby_mask[t] for t in top[:3]):
            return q, top[:3], sc[top[:3]].tolist()
    return None

res = good_query("radiology") or good_query("pathology") or good_query("medical_photograph")
q, top3, scores = res
import textwrap
qcap = imgs.caption.iloc[q]
qcap = (qcap[:220] + "…") if len(qcap) > 220 else qcap
qcap_wrapped = "\n".join(textwrap.wrap(qcap, width=30))

fig = plt.figure(figsize=(7.4, 3.0))
gs = fig.add_gridspec(1, 4, width_ratios=[1.35, 1, 1, 1], wspace=0.10)
# panel de texto (consulta)
axq = fig.add_subplot(gs[0, 0]); axq.axis("off")
axq.text(0.0, 1.0, "Consulta (texto):", fontweight="bold", va="top", fontsize=9.5)
axq.text(0.0, 0.88, f'"{qcap_wrapped}"', va="top", fontsize=7.8, family="serif")
axq.text(0.0, 0.0, "→ recuperar imágenes", va="bottom", fontsize=8.5, style="italic")
# top-3 imágenes
for rank, (t, sc) in enumerate(zip(top3, scores)):
    ax = fig.add_subplot(gs[0, rank + 1])
    ax.imshow(load_sq(imgs.path.iloc[t]))
    ax.axis("off")
    correct = (t == q)
    color = "#1a9850" if correct else "#888888"
    ax.set_title(f"#{rank+1}  cos={sc:.2f}" + ("  ✓" if correct else ""),
                 fontsize=9, color=color)
    for s in ["top", "bottom", "left", "right"]:
        ax.spines[s].set_visible(True); ax.spines[s].set_color(color)
        ax.spines[s].set_linewidth(2.5 if correct else 1.0)
fig.savefig(f"{OUT}/fig_qualitative.pdf")
plt.close(fig)
print(f"fig_qualitative.pdf OK  (query modality={imgs.image_type.iloc[q]}, top1 correcto={top3[0]==q})")
