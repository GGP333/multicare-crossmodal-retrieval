#!/usr/bin/env bash
# Pipeline completo, de los datos crudos a las figuras del manuscrito.
#
#   conda activate multicare-crossmodal
#   ./run_pipeline.sh
#
# Requiere MultiCaReDataset/ en este mismo directorio (ver README).
# Tiempo total: ~20 min en una RTX 4070 Ti SUPER.
set -euo pipefail

if [ ! -d MultiCaReDataset ]; then
  echo "Falta MultiCaReDataset/. Descargar desde https://doi.org/10.5281/zenodo.10079369" >&2
  exit 1
fi

echo "== 1/6  Análisis exploratorio =============================="
python eda_multicare.py

echo "== 2/6  Comprobación de los modelos ========================"
python validate_models.py

echo "== 3/6  Preparación de datos ==============================="
python data_prep.py

echo "== 4/6  Embeddings (CLIP y BiomedCLIP) ====================="
python embed.py

echo "== 5/6  Evaluación de recuperación ========================="
python retrieval_eval.py

echo "== 6/6  Figuras ============================================"
python make_result_figs.py
python make_example_figs.py

echo
echo "Listo. Métricas en artifacts/metrics_*.json, figuras en figs/."
