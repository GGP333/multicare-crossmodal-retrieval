"""Figuras de resultados y tabla LaTeX del manuscrito.

Uso:  python make_result_figs.py

Lee solo artifacts/metrics_*.json, así que corre en segundos y sin GPU una vez
que la evaluación ha terminado. Salidas: figs/fig_results.pdf, fig_alpha.pdf,
fig_modality.pdf, y la tabla principal por stdout.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150,
                     "savefig.bbox": "tight"})
OUT = "figs"
os.makedirs(OUT, exist_ok=True)
C_CLIP, C_BIO = "#9aa7b3", "#2c6fbb"
M = {m: json.load(open(f"artifacts/metrics_{m}.json")) for m in ["CLIP", "BiomedCLIP"]}

# ---- Fig 1: comparación de modelos por dirección (R@10 y MRR) ----
dirs = [("I2T", "I→T"), ("T2I", "T→I"), ("TI2C_concat", "TI→C")]
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))
for ax, metric, ttl in [(axes[0], "R@10", "Recall@10"), (axes[1], "MRR", "MRR")]:
    x = np.arange(len(dirs)); w = 0.38
    clip = [M["CLIP"][k][metric] for k, _ in dirs]
    bio = [M["BiomedCLIP"][k][metric] for k, _ in dirs]
    ax.bar(x - w/2, clip, w, label="CLIP", color=C_CLIP)
    ax.bar(x + w/2, bio, w, label="BiomedCLIP", color=C_BIO)
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in dirs])
    ax.set_title(ttl); ax.set_ylim(0, max(bio)*1.25)
    for xi, v in zip(x - w/2, clip): ax.text(xi, v+0.005, f"{v:.2f}", ha="center", fontsize=7.5)
    for xi, v in zip(x + w/2, bio): ax.text(xi, v+0.005, f"{v:.2f}", ha="center", fontsize=7.5)
axes[0].legend(frameon=False, fontsize=9)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_results.pdf"); plt.close(fig)

# ---- Fig 2: barrido de alpha (TI->C) ----
fig, ax = plt.subplots(figsize=(5.2, 3.3))
for m, col in [("CLIP", C_CLIP), ("BiomedCLIP", C_BIO)]:
    sw = M[m]["TI2C_alpha_sweep"]
    al = [float(a) for a in sw]; r1 = [sw[a]["R@1"] for a in sw]
    ax.plot(al, r1, "o-", color=col, label=m)
    best = max(range(len(al)), key=lambda i: r1[i])
    ax.scatter([al[best]], [r1[best]], s=120, facecolors="none", edgecolors=col, zorder=5)
ax.set_xlabel(r"$\alpha$  (0 = solo texto/caption,  1 = solo imagen)")
ax.set_ylabel("Recall@1 (TI→C)")
ax.legend(frameon=False); ax.grid(alpha=0.25)
fig.savefig(f"{OUT}/fig_alpha.pdf"); plt.close(fig)

# ---- Fig 3: rendimiento por modalidad (BiomedCLIP, I->T) ----
es = {"radiology": "Radiología", "pathology": "Patología",
      "medical_photograph": "Fotografía médica", "chart": "Gráficos",
      "ophthalmic_imaging": "Imagen oftálmica", "endoscopy": "Endoscopía",
      "electrography": "Electrografía"}
bm = M["BiomedCLIP"]["I2T"]["by_modality"]
items = sorted(bm.items(), key=lambda kv: kv[1]["R@10"])
fig, ax = plt.subplots(figsize=(5.6, 3.3))
labels = [es.get(k, k) for k, _ in items]
r10 = [v["R@10"] for _, v in items]
ax.barh(labels, r10, color=C_BIO)
for i, v in enumerate(r10): ax.text(v+0.004, i, f"{v:.2f}", va="center", fontsize=8)
ax.set_xlabel("Recall@10 (BiomedCLIP, I→T)")
ax.set_xlim(0, max(r10)*1.18)
fig.savefig(f"{OUT}/fig_modality.pdf"); plt.close(fig)

# ---- Tabla LaTeX principal ----
def row(model, key, name):
    v = M[model][key]
    return (f"{name} & {v['R@1']:.3f} & {v['R@5']:.3f} & {v['R@10']:.3f} "
            f"& {v['MRR']:.3f} & {v['nDCG@10']:.3f} \\\\")
print("% --- tabla principal ---")
for model in ["CLIP", "BiomedCLIP"]:
    print(f"\\multirow{{3}}{{*}}{{{model}}}")
    print(" & " + row(model, "I2T", "I$\\rightarrow$T")[0:])
    print(" & " + row(model, "T2I", "T$\\rightarrow$I"))
    print(" & " + row(model, "TI2C_concat", "TI$\\rightarrow$C"))
    print("\\midrule" if model == "CLIP" else "")
print("OK figs:", "fig_results.pdf fig_alpha.pdf fig_modality.pdf")
