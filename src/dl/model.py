"""Load the pretrained DistilBERT sentiment model.

This module only loads. No tokenization, no inference, no chunking.
Nothing here is trained or fine-tuned.

Public API:
    load_sentiment_model() -> (tokenizer, model, device)
"""

from __future__ import annotations

from typing import Tuple

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DistilBertForSequenceClassification,
    PreTrainedTokenizerBase,
)

# DistilBERT fine-tuned on SST-2 (binary: NEGATIVE / POSITIVE).
MODEL_CHECKPOINT = "distilbert-base-uncased-finetuned-sst-2-english"

# DistilBERT's positional embeddings stop at 512 tokens.
MAX_SEQUENCE_LENGTH = 512


def get_device() -> torch.device:
    """Return the best available device: CUDA, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_sentiment_model(
    checkpoint: str = MODEL_CHECKPOINT,
) -> Tuple[PreTrainedTokenizerBase, DistilBertForSequenceClassification, torch.device]:
    """Load the tokenizer and sentiment model onto the best available device.

    The model is returned in evaluation mode, ready for inference.

    Args:
        checkpoint: Hugging Face model ID.

    Returns:
        (tokenizer, model, device)

    Raises:
        RuntimeError: If the checkpoint cannot be loaded.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load '{checkpoint}'. Check your internet connection "
            f"for the first download, or the model name. Original error: {exc}"
        ) from exc

    device = get_device()
    model.to(device)
    model.eval()  # disables dropout; we never train this model

    return tokenizer, model, device


if __name__ == "__main__":
    tokenizer, model, device = load_sentiment_model()

    print(f"Checkpoint : {MODEL_CHECKPOINT}")
    print(f"Device     : {device}")
    print(f"Model      : {type(model).__name__}")
    print(f"Tokenizer  : {type(tokenizer).__name__}")
    print(f"Labels     : {model.config.id2label}")
    print(f"Max length : {MAX_SEQUENCE_LENGTH}")
    print(f"Parameters : {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training   : {model.training} (should be False)")