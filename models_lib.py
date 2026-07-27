"""Carga de los dos codificadores visión-lenguaje evaluados: CLIP y BiomedCLIP.

Este es el único archivo del repositorio donde se define un modelo. Todo lo demás
(embed.py, validate_models.py, el notebook de demostración) lo importa desde aquí,
de modo que los hiperparámetros de codificación —longitud de contexto, recorte de
texto— están definidos en un solo lugar.
"""
import multiprocessing as mp

try:
    # Python 3.14 usa "forkserver" por defecto, lo que rompe la carga de modelos
    # de Hugging Face dentro de los workers del DataLoader.
    mp.set_start_method("fork", force=True)
except RuntimeError:
    pass

import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from dataclasses import dataclass
from typing import Callable, List

import torch
import torch.nn.functional as F

# Se usa GPU si está disponible; si no, todo corre en CPU (más lento, pero corre).
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Dimensión del espacio de embedding compartido. Coincide en ambos modelos, lo que
# permite reutilizar el mismo código de evaluación.
DIM = 512

MODELOS = {
    # nombre        : (etiqueta open_clip, tokens de contexto, recorte de texto en caracteres)
    "CLIP":       ("ViT-B-16-quickgelu", 77, 300),
    "BiomedCLIP": ("hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224", 256, 1200),
}


@dataclass
class Backbone:
    """Modelo cargado junto con todo lo necesario para codificar imágenes y texto."""

    nombre: str
    model: torch.nn.Module
    preprocess: Callable      # PIL.Image -> tensor (3, 224, 224)
    tokenizer: Callable
    context_length: int
    max_chars: int
    device: str

    def tokenize(self, textos: List[str]) -> torch.Tensor:
        """Recorta y tokeniza. El recorte previo evita tokenizar texto que el
        codificador descartaría igualmente por su límite de contexto."""
        return self.tokenizer([t[: self.max_chars] for t in textos],
                              context_length=self.context_length)


def load(nombre: str, device: str = None) -> Backbone:
    """Carga uno de los modelos de MODELOS. Los pesos se descargan de Hugging Face
    la primera vez y quedan en la caché local (~/.cache/huggingface)."""
    import open_clip

    if nombre not in MODELOS:
        raise ValueError(f"Modelo no soportado: {nombre!r}. Opciones: {list(MODELOS)}")
    tag, ctx, max_chars = MODELOS[nombre]
    device = device or DEVICE

    if nombre == "CLIP":
        model, _, preprocess = open_clip.create_model_and_transforms(tag, pretrained="openai")
    else:
        model, preprocess = open_clip.create_model_from_pretrained(tag)
    tokenizer = open_clip.get_tokenizer(tag)

    model = model.to(device).eval()
    return Backbone(nombre, model, preprocess, tokenizer, ctx, max_chars, device)


def load_encoders(nombre: str, device: str = None):
    """Devuelve (enc_img, enc_txt), la interfaz cómoda para uso interactivo.

    enc_img: lista de PIL.Image -> tensor (n, 512) en CPU, L2-normalizado
    enc_txt: lista de str       -> tensor (n, 512) en CPU, L2-normalizado

    Al estar L2-normalizados, la similitud coseno se reduce a un producto interno.
    Para volúmenes grandes conviene usar embed.py, que además hace batching y fp16.
    """
    bb = load(nombre, device)

    @torch.no_grad()
    def enc_img(pils):
        x = torch.stack([bb.preprocess(im) for im in pils]).to(bb.device)
        return F.normalize(bb.model.encode_image(x).float()).cpu()

    @torch.no_grad()
    def enc_txt(textos):
        toks = bb.tokenize(textos).to(bb.device)
        return F.normalize(bb.model.encode_text(toks).float()).cpu()

    return enc_img, enc_txt


# Atajos usados por el notebook de demostración.
def load_clip(device=None):
    return load_encoders("CLIP", device)


def load_biomedclip(device=None):
    return load_encoders("BiomedCLIP", device)
