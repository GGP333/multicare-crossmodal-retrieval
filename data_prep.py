"""Preparación de datos: aplana el corpus MultiCaRe en dos índices alineados por
fila con los embeddings, y muestrea las consultas de evaluación.

Uso:  python data_prep.py

Salidas en artifacts/
  cases_index.parquet   93.816 casos: texto, tópicos, modalidad dominante
  images_index.parquet  130.791 imágenes: ruta, leyenda, caso al que pertenecen
  q_*.npy               muestras estratificadas de consultas (ver más abajo)
  prep_summary.json     recuentos para el manuscrito

Las semillas están fijadas (42 global; 1, 2 y 3 para los tres muestreos), de modo
que el pipeline es determinista de extremo a extremo.
"""
import json
import os

import numpy as np
import pandas as pd

DATA = "MultiCaReDataset"
N_QUERIES = 5000

os.makedirs("artifacts", exist_ok=True)
np.random.seed(42)

# ----------------------------------------------------------------------
# 1. Casos clínicos. cases.parquet viene anidado: una fila por artículo, con
#    una lista de casos dentro; se aplana a una fila por caso.
# ----------------------------------------------------------------------
cases_raw = pd.read_parquet(f"{DATA}/cases.parquet")
filas = [(c.get("case_id"), art, c.get("case_text") or "")
         for art, lista in zip(cases_raw.article_id, cases_raw.cases)
         for c in lista]
cases = pd.DataFrame(filas, columns=["case_id", "article_id", "case_text"])
cases = cases[cases.case_text.str.len() > 0].reset_index(drop=True)
print(f"casos con texto: {len(cases):,}")

# ----------------------------------------------------------------------
# 2. Tópicos por artículo (MeSH mayor + keywords).
#    Se usan para la relevancia graduada del nDCG: dos casos que comparten al
#    menos un tópico se consideran parcialmente relevantes entre sí (rel = 1).
#    Los términos genéricos se descartan porque los comparte casi todo el corpus
#    y harían que cualquier par de casos pareciera relacionado.
# ----------------------------------------------------------------------
GENERICOS = {"humans", "case reports", "male", "female", "adult", "aged",
             "middle aged", "young adult", "adolescent", "child", "infant",
             "child, preschool", "aged, 80 and over", "infant, newborn",
             "case report", "treatment outcome", "follow-up studies",
             "retrospective studies", "research support, non-u.s. gov't",
             "review", "letter", "english abstract"}


def como_lista(x):
    if x is None:
        return []
    if isinstance(x, np.ndarray):
        return list(x)
    return list(x) if isinstance(x, (list, tuple)) else [x]


topicos_por_articulo = {}
for d in pd.read_parquet(f"{DATA}/metadata.parquet").article_metadata:
    toks = {str(m).lower().strip() for m in como_lista(d.get("major_mesh_terms"))}
    if not toks:  # si el artículo no tiene MeSH "mayor", se usan los normales
        toks = {str(m).lower().strip() for m in como_lista(d.get("mesh_terms"))}
    toks |= {str(k).lower().strip() for k in como_lista(d.get("keywords"))}
    topicos_por_articulo[d.get("pmcid")] = {t for t in toks
                                            if t and t not in GENERICOS and len(t) > 2}

cases["topics"] = cases.article_id.map(lambda a: topicos_por_articulo.get(a, set()))

# ----------------------------------------------------------------------
# 3. Imágenes. En captions_and_labels.csv, patient_id es el identificador del caso.
#    La ruta se deriva del nombre de archivo: PMC1234567_fig1.webp -> PMC1/PMC12/…
# ----------------------------------------------------------------------
cl = pd.read_csv(f"{DATA}/captions_and_labels.csv",
                 usecols=["file", "patient_id", "caption", "image_type",
                          "image_subtype", "radiology_region"])
cl = cl.rename(columns={"patient_id": "case_id"})
cl["path"] = cl.file.map(lambda fn: os.path.join(DATA, fn[:4], fn[:5], fn))
print(f"imágenes: {len(cl):,}")

# Índice entero de caso: es lo que enlaza cada imagen con su fila en cases.
posicion_caso = {cid: i for i, cid in enumerate(cases.case_id)}
cl["case_idx"] = cl.case_id.map(posicion_caso)
cl = cl[cl.case_idx.notna()].reset_index(drop=True)
cl["case_idx"] = cl.case_idx.astype(int)
print(f"imágenes cuyo caso tiene texto: {len(cl):,}")

# Modalidad dominante y número de imágenes por caso (para estratificar el muestreo).
dominante = cl.groupby("case_idx").image_type.agg(lambda s: s.value_counts().index[0])
cases["dom_image_type"] = cases.index.map(dominante).fillna("none")
cases["n_images"] = cases.index.map(cl.groupby("case_idx").size()).fillna(0).astype(int)


# ----------------------------------------------------------------------
# 4. Muestreo estratificado de consultas.
#    Se estratifica por modalidad de imagen para que la muestra conserve la
#    composición del corpus: sin esto, las modalidades minoritarias (endoscopía,
#    electrografía) quedarían con muy pocas consultas y su métrica sería ruido.
# ----------------------------------------------------------------------
def muestreo_estratificado(df, columna, n, semilla):
    rng = np.random.RandomState(semilla)
    grupos = df.groupby(columna)
    fracciones = grupos.size() / len(df)
    elegidos = []
    for g, indices in grupos.groups.items():
        indices = list(indices)
        k = min(max(1, int(round(fracciones[g] * n))), len(indices))
        elegidos.extend(rng.choice(indices, size=k, replace=False).tolist())
    rng.shuffle(elegidos)
    return elegidos[:n]


# El conjunto que la evaluación usa realmente en las tres direcciones: cada fila-imagen
# aporta a la vez la imagen, su leyenda y su caso, así que una sola muestra basta.
q_imagenes = muestreo_estratificado(cl, "image_type", N_QUERIES, semilla=1)
# Muestras de los diseños alternativos considerados durante el desarrollo (consultas
# a nivel de caso). Se conservan por trazabilidad; retrieval_eval.py no las usa.
q_casos = muestreo_estratificado(cases[cases.n_images > 0], "dom_image_type", N_QUERIES, semilla=2)
q_imagenes_alt = muestreo_estratificado(cl, "image_type", N_QUERIES, semilla=3)
print(f"consultas muestreadas: {len(q_imagenes)}")

# ----------------------------------------------------------------------
# 5. Guardar
# ----------------------------------------------------------------------
salida = cases[["case_id", "article_id", "case_text", "dom_image_type", "n_images"]].copy()
salida["topics"] = cases.topics.map(list)  # parquet no admite set
salida.to_parquet("artifacts/cases_index.parquet")
cl[["file", "path", "case_idx", "image_type", "image_subtype",
    "radiology_region", "caption"]].to_parquet("artifacts/images_index.parquet")

np.save("artifacts/q_it_img.npy", np.array(q_imagenes))
np.save("artifacts/q_ti_case.npy", np.array(q_casos))
np.save("artifacts/q_tic_img.npy", np.array(q_imagenes_alt))

resumen = {
    "n_cases_text": int(len(cases)),
    "n_cases_with_img": int((cases.n_images > 0).sum()),
    "n_images": int(len(cl)),
    "n_queries_each": N_QUERIES,
    "image_type_dist_queries": cl.loc[q_imagenes].image_type.value_counts().to_dict(),
    "median_text_len": int(cases.case_text.str.len().median()),
}
json.dump(resumen, open("artifacts/prep_summary.json", "w"), indent=2)
print(json.dumps(resumen, indent=2, ensure_ascii=False))
print("OK data_prep")
