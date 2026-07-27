"""Análisis exploratorio del dataset MultiCaRe.

Uso:  python eda_multicare.py

Salidas: artifacts/eda_stats.json y 7 figuras en figs/. No requiere GPU ni los
embeddings —solo lee el corpus—, así que además sirve para comprobar que el
dataset quedó bien descomprimido antes de invertir tiempo de cómputo.
"""
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

DATA = "MultiCaReDataset"
OUT = "figs"
os.makedirs(OUT, exist_ok=True)
os.makedirs("artifacts", exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})

stats = {}

# ----------------------------------------------------------------------
# 1. CASOS CLÍNICOS  (cases.parquet -> lista anidada)
# ----------------------------------------------------------------------
cases_raw = pd.read_parquet(f"{DATA}/cases.parquet")
rows = []
for art_id, lst in zip(cases_raw.article_id, cases_raw.cases):
    for c in lst:
        rows.append({
            "article_id": art_id,
            "case_id": c.get("case_id"),
            "age": c.get("age"),
            "gender": c.get("gender"),
            "text_len": len(c.get("case_text") or ""),
        })
cases = pd.DataFrame(rows)
stats["n_cases"] = len(cases)
stats["n_articles"] = cases.article_id.nunique()

# Género
g = cases.gender.fillna("Unknown").value_counts()
stats["gender_counts"] = g.to_dict()
stats["gender_pct"] = (g / len(cases) * 100).round(1).to_dict()

# Edad
age = cases.age.dropna()
stats["age_median"] = float(age.median())
stats["age_q1"] = float(age.quantile(0.25))
stats["age_q3"] = float(age.quantile(0.75))
stats["age_min"] = float(age.min())
stats["age_max"] = float(age.max())
stats["age_n"] = int(age.notna().sum())

# Longitud de texto
tl = cases.text_len
stats["text_median"] = int(tl.median())
stats["text_q1"] = int(tl.quantile(0.25))
stats["text_q3"] = int(tl.quantile(0.75))
stats["text_min"] = int(tl.min())
stats["text_max"] = int(tl.max())
stats["text_mean"] = round(float(tl.mean()), 1)

# Casos por artículo
cpa = cases.groupby("article_id").size()
stats["cases_per_article_median"] = float(cpa.median())
stats["cases_per_article_max"] = int(cpa.max())
stats["cases_per_article_mean"] = round(float(cpa.mean()), 2)

# ----------------------------------------------------------------------
# 2. IMÁGENES  (captions_and_labels.csv)
# ----------------------------------------------------------------------
cl = pd.read_csv(f"{DATA}/captions_and_labels.csv")
stats["n_images"] = len(cl)
stats["n_main_images"] = cl.main_image.nunique()
stats["n_patients_img"] = cl.patient_id.nunique()

it = cl.image_type.value_counts()
stats["image_type_counts"] = it.to_dict()
stats["image_type_pct"] = (it / len(cl) * 100).round(1).to_dict()

sub = cl.image_subtype.value_counts()
stats["image_subtype_top"] = sub.head(12).to_dict()

reg = cl.radiology_region.value_counts()
stats["radiology_region"] = reg.to_dict()

view = cl.radiology_view.value_counts()
stats["radiology_view_top"] = view.head(8).to_dict()

# Longitud de caption
cl["cap_len"] = cl.caption.fillna("").str.len()
stats["caption_median"] = int(cl.cap_len.median())
stats["caption_mean"] = round(float(cl.cap_len.mean()), 1)

# Licencias
lic = cl.license.value_counts()
stats["license_counts"] = lic.to_dict()

# ----------------------------------------------------------------------
# 3. MULTIMODALIDAD: imágenes por caso y tipos por paciente
# ----------------------------------------------------------------------
# Imágenes por paciente (patient_id en captions == case)
img_per_patient = cl.groupby("patient_id").size()
stats["img_per_case_median"] = float(img_per_patient.median())
stats["img_per_case_max"] = int(img_per_patient.max())
stats["img_per_case_mean"] = round(float(img_per_patient.mean()), 2)

# Nº de tipos de imagen distintos por paciente
types_per_patient = cl.groupby("patient_id").image_type.nunique()
stats["multitype_patients"] = int((types_per_patient > 1).sum())
stats["multitype_pct"] = round((types_per_patient > 1).mean() * 100, 1)
stats["n_patients_total"] = int(types_per_patient.shape[0])

# ----------------------------------------------------------------------
# 4. METADATOS (metadata.parquet -> dict anidado)
# ----------------------------------------------------------------------
meta_raw = pd.read_parquet(f"{DATA}/metadata.parquet")
def aslist(x):
    if x is None:
        return []
    if isinstance(x, np.ndarray):
        return list(x)
    return list(x) if isinstance(x, (list, tuple)) else [x]

mrows = []
for d in meta_raw.article_metadata:
    mesh = aslist(d.get("mesh_terms"))
    kw = aslist(d.get("keywords"))
    mrows.append({
        "year": d.get("year"),
        "journal": d.get("journal"),
        "license": d.get("license"),
        "n_mesh": len(mesh),
        "n_kw": len(kw),
        "mesh": mesh,
        "kw": kw,
    })
meta = pd.DataFrame(mrows)
meta["year_num"] = pd.to_numeric(meta.year, errors="coerce")
yr = meta.year_num.dropna().astype(int)
stats["year_min"] = int(yr.min())
stats["year_max"] = int(yr.max())
year_counts = yr.value_counts().sort_index()
stats["year_counts"] = year_counts.to_dict()

# Revistas
jc = meta.journal.value_counts()
stats["n_journals"] = int(meta.journal.nunique())
stats["top_journals"] = jc.head(10).to_dict()

# MeSH y keywords agregados
from collections import Counter
mesh_counter = Counter()
for m in meta.mesh:
    mesh_counter.update(m)
kw_counter = Counter()
for k in meta.kw:
    kw_counter.update([w.lower() for w in k])
stats["total_mesh_assign"] = sum(mesh_counter.values())
stats["unique_mesh"] = len(mesh_counter)
stats["total_kw_assign"] = sum(kw_counter.values())
stats["unique_kw"] = len(kw_counter)
generic_mesh = {"Case Reports", "Humans", "Male", "Female", "Adult",
                "Middle Aged", "Aged", "Young Adult", "Adolescent",
                "Child", "Aged, 80 and over", "Infant", "Child, Preschool"}
stats["top_mesh"] = {k: v for k, v in mesh_counter.most_common(40)
                     if k not in generic_mesh}
stats["top_mesh"] = dict(list(stats["top_mesh"].items())[:12])
stats["top_kw"] = dict(kw_counter.most_common(15))

# ======================================================================
#  FIGURAS
# ======================================================================
PAL = "#2c6fbb"
PAL2 = "#bb452c"

def thousands(x, pos):
    return f"{int(x):,}".replace(",", ".")

# --- FIG A: distribución de edad por género (un panel) ---
fig, ax = plt.subplots(figsize=(6.5, 3.6))
bins = np.arange(0, 101, 5)
for gg, col in [("Female", "#bb452c"), ("Male", "#2c6fbb")]:
    sub_age = cases.loc[cases.gender == gg, "age"].dropna()
    ax.hist(sub_age, bins=bins, alpha=0.6, label=gg, color=col, edgecolor="white", linewidth=0.4)
ax.set_xlabel("Edad (años)")
ax.set_ylabel("Número de casos")
ax.yaxis.set_major_formatter(FuncFormatter(thousands))
ax.legend(title="Género", frameon=False)
fig.savefig(f"{OUT}/fig_age.pdf")
plt.close(fig)

# --- FIG B: tipos de imagen (barras horizontales) ---
fig, ax = plt.subplots(figsize=(6.5, 3.4))
labels_es = {
    "radiology": "Radiología", "pathology": "Patología",
    "medical_photograph": "Fotografía médica", "chart": "Gráficos",
    "ophthalmic_imaging": "Imagen oftálmica", "endoscopy": "Endoscopía",
    "electrography": "Electrografía",
}
it_sorted = it.sort_values()
ax.barh([labels_es[k] for k in it_sorted.index], it_sorted.values, color=PAL)
for i, v in enumerate(it_sorted.values):
    ax.text(v + 800, i, f"{v:,}".replace(",", "."), va="center", fontsize=9)
ax.set_xlabel("Número de imágenes")
ax.xaxis.set_major_formatter(FuncFormatter(thousands))
ax.set_xlim(0, it_sorted.max() * 1.18)
fig.savefig(f"{OUT}/fig_imgtype.pdf")
plt.close(fig)

# --- FIG C: subtipos de imagen top-12 ---
fig, ax = plt.subplots(figsize=(6.5, 4.0))
sub12 = sub.head(12).sort_values()
ax.barh(sub12.index, sub12.values, color=PAL2)
ax.set_xlabel("Número de imágenes")
ax.xaxis.set_major_formatter(FuncFormatter(thousands))
fig.savefig(f"{OUT}/fig_subtype.pdf")
plt.close(fig)

# --- FIG D: publicaciones por año ---
fig, ax = plt.subplots(figsize=(6.5, 3.4))
yc = year_counts[year_counts.index >= 1990]
ax.bar(yc.index, yc.values, color=PAL, width=0.85)
ax.set_xlabel("Año de publicación")
ax.set_ylabel("Número de artículos")
ax.yaxis.set_major_formatter(FuncFormatter(thousands))
fig.savefig(f"{OUT}/fig_year.pdf")
plt.close(fig)

# --- FIG E: imágenes por caso (histograma) ---
fig, ax = plt.subplots(figsize=(6.5, 3.4))
capn = img_per_patient.clip(upper=15)
ax.hist(capn, bins=np.arange(1, 17) - 0.5, color=PAL, edgecolor="white")
ax.set_xlabel("Número de imágenes por caso")
ax.set_ylabel("Número de casos")
ax.yaxis.set_major_formatter(FuncFormatter(thousands))
ax.set_xticks(range(1, 16))
fig.savefig(f"{OUT}/fig_imgpercase.pdf")
plt.close(fig)

# --- FIG F: longitud del texto clínico (histograma, recortado) ---
fig, ax = plt.subplots(figsize=(6.5, 3.4))
tl_clip = tl.clip(upper=12000)
ax.hist(tl_clip, bins=50, color=PAL2, edgecolor="white", linewidth=0.3)
ax.axvline(tl.median(), color="black", linestyle="--", linewidth=1,
           label=f"Mediana = {int(tl.median()):,}".replace(",", "."))
ax.set_xlabel("Longitud del texto clínico (caracteres)")
ax.set_ylabel("Número de casos")
ax.yaxis.set_major_formatter(FuncFormatter(thousands))
ax.legend(frameon=False)
fig.savefig(f"{OUT}/fig_textlen.pdf")
plt.close(fig)

# --- FIG G: regiones anatómicas radiología ---
fig, ax = plt.subplots(figsize=(6.5, 3.4))
reg_es = {"head": "Cabeza", "thorax": "Tórax", "abdomen": "Abdomen",
          "lower_limb": "Extr. inferior", "neck": "Cuello", "pelvis": "Pelvis",
          "upper_limb": "Extr. superior", "breast": "Mama", "whole_body": "Cuerpo entero"}
reg_s = reg.sort_values()
ax.barh([reg_es.get(k, k) for k in reg_s.index], reg_s.values, color=PAL)
ax.set_xlabel("Número de imágenes radiológicas")
ax.xaxis.set_major_formatter(FuncFormatter(thousands))
fig.savefig(f"{OUT}/fig_region.pdf")
plt.close(fig)

# ----------------------------------------------------------------------
# Guardar stats
# ----------------------------------------------------------------------
def conv(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    return str(o)

with open("artifacts/eda_stats.json", "w") as f:
    json.dump(stats, f, indent=2, ensure_ascii=False, default=conv)

# Imprimir resumen legible
print("="*60)
print("RESUMEN EDA MultiCaRe")
print("="*60)
for k, v in stats.items():
    if isinstance(v, dict) and len(v) > 6:
        print(f"{k}:")
        for kk, vv in list(v.items())[:12]:
            print(f"    {kk}: {vv}")
    else:
        print(f"{k}: {v}")
