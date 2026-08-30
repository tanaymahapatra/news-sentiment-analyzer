"""Run DistilBERT sentiment inference over long article text.

Handles articles longer than DistilBERT's 512-token limit by splitting them
into overlapping chunks, scoring each chunk, and averaging the resulting
probability distributions into one article-level prediction.

The model is loaded elsewhere (src/dl/model.py) and passed in.

Public API:
    predict_sentiment(text, tokenizer, model, device) -> SentimentResult
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import torch

from .model import MAX_SEQUENCE_LENGTH

# [CLS] and [SEP] occupy two of the 512 positions.
MAX_CONTENT_TOKENS = MAX_SEQUENCE_LENGTH - 2  # 510
CHUNK_OVERLAP = 50  # tokens shared between chunks
MIN_CHUNK_TOKENS = 64  # avoid a tiny trailing chunk
BATCH_SIZE = 8


class InferenceError(Exception):
    """Raised when the input cannot be scored."""


@dataclass
class SentimentResult:
    """Article-level sentiment prediction."""

    sentiment: str
    confidence: float
    positive_probability: float
    negative_probability: float
    num_chunks: int
    total_tokens: int
    chunk_predictions: List[Dict[str, object]] = field(default_factory=list)


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------


def chunk_token_ids(
    token_ids: List[int],
    chunk_size: int = MAX_CONTENT_TOKENS,
    overlap: int = CHUNK_OVERLAP,
) -> List[List[int]]:
    """Split token ids into overlapping windows that fit DistilBERT.

    The window slides forward by (chunk_size - overlap), so consecutive
    chunks share `overlap` tokens and no sentence is split without context
    on at least one side.

    If the final window is a tiny fragment, it is dropped and the previous
    window is slid forward to end on the last token. That keeps full coverage
    without adding a second near-identical chunk that would be double-counted
    during aggregation.
    """
    if not token_ids:
        return []
    if len(token_ids) <= chunk_size:
        return [token_ids]

    step = chunk_size - overlap
    chunks = [token_ids[i : i + chunk_size] for i in range(0, len(token_ids), step)]

    # Drop windows whose content is already fully covered by the previous one.
    chunks = [c for c in chunks if len(c) > overlap]

    if len(chunks) > 1 and len(chunks[-1]) < MIN_CHUNK_TOKENS:
        chunks.pop()  # discard the fragment
        tail_window = token_ids[-chunk_size:]  # window ending on the last token
        if len(chunks) > 1:
            chunks[-1] = tail_window  # slide the last window forward
        else:
            # Only one full window exists, so the tail cannot be absorbed
            # without losing the opening tokens. Keep both.
            chunks.append(tail_window)

    return chunks


def _build_batch(chunks: List[List[int]], tokenizer) -> Dict[str, torch.Tensor]:
    """Add special tokens, pad to the longest chunk, build the attention mask."""
    sequences = [
        [tokenizer.cls_token_id] + chunk + [tokenizer.sep_token_id] for chunk in chunks
    ]
    longest = max(len(seq) for seq in sequences)

    input_ids, attention_mask = [], []
    for seq in sequences:
        padding = longest - len(seq)
        input_ids.append(seq + [tokenizer.pad_token_id] * padding)
        attention_mask.append([1] * len(seq) + [0] * padding)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


# --------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------


def _score_chunks(chunks: List[List[int]], tokenizer, model, device) -> torch.Tensor:
    """Return a (num_chunks, num_labels) tensor of class probabilities."""
    all_probs = []

    with torch.inference_mode():
        for start in range(0, len(chunks), BATCH_SIZE):
            batch = _build_batch(chunks[start : start + BATCH_SIZE], tokenizer)
            batch = {key: value.to(device) for key, value in batch.items()}

            logits = model(**batch).logits
            probs = torch.softmax(logits, dim=-1)
            all_probs.append(probs.cpu())

    return torch.cat(all_probs, dim=0)


def predict_sentiment(text: str, tokenizer, model, device) -> SentimentResult:
    """Predict article-level sentiment from cleaned text.

    Args:
        text: Cleaned article text.
        tokenizer, model, device: As returned by load_sentiment_model().

    Returns:
        SentimentResult with the label, confidence, per-class probabilities
        and per-chunk detail.

    Raises:
        TypeError: If text is not a string.
        InferenceError: If the text contains no tokens to score.
    """
    if not isinstance(text, str):
        raise TypeError(
            f"predict_sentiment() expects a string, got {type(text).__name__}."
        )
    if not text.strip():
        raise InferenceError("Cannot analyze empty text.")

    # verbose=False: the tokenizer would otherwise warn that the text exceeds
    # 512 tokens. That is expected here -- chunking is handled below.
    token_ids = tokenizer(text, add_special_tokens=False, verbose=False)["input_ids"]
    if not token_ids:
        raise InferenceError("Text contains no recognizable tokens.")

    chunks = chunk_token_ids(token_ids)
    probs = _score_chunks(chunks, tokenizer, model, device)

    # Article-level distribution = mean of the chunk distributions.
    mean_probs = probs.mean(dim=0)

    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
    best_index = int(mean_probs.argmax())
    by_label = {label: float(mean_probs[i]) for i, label in enumerate(labels)}

    chunk_predictions = [
        {
            "chunk": i + 1,
            "tokens": len(chunks[i]),
            "label": labels[int(row.argmax())],
            "confidence": round(float(row.max()), 4),
        }
        for i, row in enumerate(probs)
    ]

    return SentimentResult(
        sentiment=labels[best_index],
        confidence=round(float(mean_probs[best_index]), 4),
        positive_probability=round(by_label.get("POSITIVE", 0.0), 4),
        negative_probability=round(by_label.get("NEGATIVE", 0.0), 4),
        num_chunks=len(chunks),
        total_tokens=len(token_ids),
        chunk_predictions=chunk_predictions,
    )
