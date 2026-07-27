# Recuperación cross-modal de casos clínicos con texto e imágenes médicas

Código del trabajo *"Recuperación Cross-Modal de Casos Clínicos Mediante Texto e Imágenes
Médicas: Un Enfoque Multimodal sobre el Dataset MultiCaRe"*.

**Gabriel Guerra Pinto** — Doctorado en Ciencias e Ingeniería para la Salud,
Universidad de Valparaíso. Curso *Fundamentos de Ciencia de Datos*, 2026.

## Qué hace

Sobre [MultiCaRe](https://doi.org/10.5281/zenodo.10079369) (93.816 casos clínicos y 130.791
imágenes médicas de reportes de acceso abierto), compara dos modelos de visión-lenguaje en
recuperación *zero-shot*: **CLIP** (ViT-B/16, imágenes generales) y **BiomedCLIP** (ViT-B/16 +
PubMedBERT, literatura biomédica), en tres direcciones de búsqueda:

| Dirección | Consulta | Se recupera |
|---|---|---|
| **I→T** | una imagen médica | la leyenda que la describe |
| **T→I** | una descripción en texto | la imagen correspondiente |
| **TI→C** | imagen + texto a la vez | el caso clínico completo |

La pregunta de fondo: cuánto aporta preentrenar en el dominio biomédico, y cuánto aporta
combinar las dos modalidades frente a usar solo una.

## Resultados

Corpus completo, 4.999 consultas, sin ajuste fino. TI→C con fusión por concatenación y
evaluación *leave-one-out*.

| Modelo | Dirección | R@1 | R@5 | R@10 | MRR | nDCG@10 |
|---|---|---|---|---|---|---|
| CLIP | I→T | 0,008 | 0,019 | 0,025 | 0,016 | 0,014 |
| CLIP | T→I | 0,008 | 0,017 | 0,024 | 0,014 | 0,013 |
| CLIP | TI→C | 0,044 | 0,079 | 0,102 | 0,065 | 0,051 |
| **BiomedCLIP** | I→T | **0,071** | **0,166** | **0,223** | **0,122** | **0,103** |
| **BiomedCLIP** | T→I | 0,070 | 0,160 | 0,210 | 0,119 | 0,102 |
| **BiomedCLIP** | TI→C | **0,114** | **0,212** | **0,263** | **0,166** | **0,122** |

1. **El dominio importa más que la arquitectura.** BiomedCLIP supera a CLIP por un factor
   cercano a nueve en R@10 pese a compartir el mismo codificador visual: la diferencia está en
   los datos de preentrenamiento.
2. **La fusión multimodal supera a cualquier modalidad aislada.** El óptimo del peso de fusión
   está en α = 0,75 (R@1 = 0,139), por encima del extremo textual (α = 0: 0,044) y del visual
   (α = 1: 0,125).
3. **El rendimiento depende fuertemente de la modalidad de imagen**, lo que hace insuficiente
   reportar solo la métrica agregada.

Desglose completo en [`artifacts/metrics_CLIP.json`](artifacts/metrics_CLIP.json) y
[`artifacts/metrics_BiomedCLIP.json`](artifacts/metrics_BiomedCLIP.json).

Sobre la magnitud de las cifras: recuperar *la* leyenda exacta entre 130.791 candidatas es una
tarea deliberadamente estricta, y un R@1 de 0,07 no significa fallar el 93 % de las veces desde
el punto de vista clínico — los resultados "incorrectos" suelen ser casos genuinamente
similares.

## Demostración

[`notebooks/demo_recuperacion.ipynb`](notebooks/demo_recuperacion.ipynb) está guardado **con sus
salidas**.
Muestra consultas en texto libre, un acierto y un fallo de I→T analizados en paralelo, una
consulta multimodal que recupera el caso correcto donde la leyenda sola falla, y la comparación
CLIP vs. BiomedCLIP.

## Uso

Requisitos: Python 3.14, ~10 GB de disco. GPU NVIDIA recomendada (~20 min el pipeline completo
en una RTX 4070 Ti SUPER); en CPU corre igual, más lento.

```bash
conda env create -f environment.yml && conda activate multicare-crossmodal
# alternativa: python -m venv .venv && .venv/bin/pip install -r requirements.txt
```

Ambas rutas instalan PyTorch para CUDA 13.0; para CPU u otra versión, ajustar el índice según
<https://pytorch.org/get-started/locally/>. Los pesos de CLIP y BiomedCLIP (~1 GB) se descargan
solos desde Hugging Face.

El corpus no se versiona (2,9 GB): descargarlo de
[Zenodo](https://doi.org/10.5281/zenodo.10079369) y descomprimirlo como `MultiCaReDataset/` en
la raíz. Después:

```bash
./run_pipeline.sh           # todo de una vez, o paso a paso:
python eda_multicare.py     # 1. exploración del corpus → eda_stats.json + 7 figuras
python validate_models.py   # 2. comprobar que los modelos cargan y discriminan  (~30 s)
python data_prep.py         # 3. índices y muestreo de consultas                 (~3 min)
python embed.py             # 4. embeddings de imágenes, leyendas y narrativas   (~8 min, GPU)
python retrieval_eval.py    # 5. evaluación → metrics_*.json                     (~5 min, GPU)
python make_result_figs.py  # 6. figuras de resultados + tabla LaTeX             (~10 s)
python make_example_figs.py # 7. figuras con imágenes de ejemplo                 (~1 min)
```

Los pasos 2, 4 y 5 procesan los dos modelos, o solo uno si se indica: `python embed.py
BiomedCLIP`. `models_lib.py` es el único lugar donde se define un modelo.

## Decisiones metodológicas

- **Se evalúa contra las leyendas, no contra las narrativas.** La narrativa de un caso contiene
  el texto de sus propias leyendas: buscar la narrativa más parecida a una imagen sería en parte
  una búsqueda de texto contra sí mismo, una fuga que inflaría las métricas.
- **Leave-one-out en TI→C.** Un caso se representa promediando los embeddings de sus imágenes;
  si la imagen de consulta quedara en ese promedio, el caso correcto contendría literalmente la
  consulta. Se recalcula el caso objetivo excluyéndola.
- **Relevancia graduada.** Recall y MRR solo premian el acierto exacto; el nDCG@10 asigna rel = 2
  al acierto exacto y rel = 1 a lo parcialmente relevante (otra imagen del mismo caso, u otro
  caso con tópicos MeSH compartidos), más cerca de lo que un clínico consideraría útil.
- **Modelos descartados.** El diseño original incluía PMC-CLIP y MedCLIP. Los puntos de control
  de PMC-CLIP ya no están accesibles; MedCLIP carga, pero su codificador visual produce
  representaciones degeneradas (similitud coseno media de 0,92 entre imágenes arbitrarias, frente
  a 0,36 en BiomedCLIP), así que se excluyó. `validate_models.py` es la comprobación que lo
  detectó.

## Limitaciones

- La evaluación es *zero-shot*: los resultados son una cota inferior de lo alcanzable con ajuste
  fino sobre el dominio.
- El componente textual de TI→C es una leyenda, no la narrativa completa: BiomedCLIP admite 256
  tokens, muy por debajo de la mediana de 2.526 caracteres de las narrativas.
- Las métricas se reportan sin intervalos de confianza.
- La búsqueda es por producto interno exacto; un corpus mayor requeriría un índice aproximado
  (FAISS, HNSW).

## Referencia

Nievas Offidani, M. (2025). *MultiCaRe: An open-source clinical case dataset for medical image
classification and multimodal AI applications*. Data in Brief.
<https://doi.org/10.5281/zenodo.10079369>
