"""Evaluate the pretrained DistilBERT checkpoint on Financial PhraseBank.

Nothing is trained here. The same pipeline used by the Streamlit app
(clean_text -> predict_sentiment) is run over labeled data.

    python -m src.evaluation.evaluate_dl

Binary metrics come from the positive/negative subset only. Neutral examples
are reported separately because the checkpoint has no neutral class.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from src.dl.inference import predict_sentiment
from src.dl.model import MODEL_CHECKPOINT, load_sentiment_model
from src.scraper.text_cleaner import clean_text

DATA_DIR = Path("data/evaluation")
BINARY_CSV = DATA_DIR / "financial_phrasebank_binary.csv"
FULL_CSV = DATA_DIR / "financial_phrasebank_full.csv"
BINARY_RESULTS_CSV = DATA_DIR / "dl_results.csv"
NEUTRAL_RESULTS_CSV = DATA_DIR / "neutral_results.csv"

LABEL_ORDER = ["NEGATIVE", "POSITIVE"]
PROGRESS_EVERY = 250


# --------------------------------------------------------------------------
# Loading and validation
# --------------------------------------------------------------------------


def load_csv(path: Path, required_columns: Tuple[str, ...]) -> pd.DataFrame:
    """Load a prepared CSV and check that the expected columns exist."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run:  python -m src.evaluation.prepare_dataset"
        )

    frame = pd.read_csv(path)
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing required column(s): {missing}")
    if frame.empty:
        raise ValueError(f"{path} contains no rows.")

    return frame


def check_labels(frame: pd.DataFrame, allowed: set, path: Path) -> None:
    """Fail if the file contains labels we did not expect."""
    found = set(frame["true_label"].unique())
    if not found.issubset(allowed):
        raise ValueError(
            f"{path} contains unexpected labels {found - allowed}. "
            f"Expected only {allowed}."
        )


# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------


def run_predictions(
    texts: List[str], tokenizer, model, device, description: str
) -> Tuple[List[dict], float]:
    """Score every text with the app's pipeline. Returns (rows, seconds).

    Any failure raises with the offending row index. Examples are never
    silently skipped, because a shrinking denominator would distort metrics.
    """
    print(f"\n{description}: {len(texts)} examples")
    rows: List[dict] = []

    start = time.perf_counter()
    for index, text in enumerate(texts, start=1):
        try:
            result = predict_sentiment(clean_text(text), tokenizer, model, device)
        except Exception as error:
            raise RuntimeError(
                f"Prediction failed on row {index - 1}: {type(error).__name__}: {error}"
            ) from error

        rows.append(
            {
                "predicted_label": result.sentiment,
                "confidence": result.confidence,
                "positive_probability": result.positive_probability,
                "negative_probability": result.negative_probability,
            }
        )

        if index % PROGRESS_EVERY == 0 or index == len(texts):
            print(f"  {index}/{len(texts)}")

    return rows, time.perf_counter() - start


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def report_binary(frame: pd.DataFrame, elapsed: float) -> None:
    """Print accuracy, per-class metrics and the confusion matrix."""
    true = frame["true_label"].tolist()
    predicted = frame["predicted_label"].tolist()

    accuracy = accuracy_score(true, predicted)
    macro = precision_recall_fscore_support(
        true, predicted, labels=LABEL_ORDER, average="macro", zero_division=0
    )
    per_class = precision_recall_fscore_support(
        true, predicted, labels=LABEL_ORDER, average=None, zero_division=0
    )

    print("\n" + "=" * 62)
    print("BINARY EVALUATION (positive/negative subset only)")
    print("=" * 62)
    print(f"Examples evaluated : {len(frame)}")
    print(f"Class distribution : " f"{frame['true_label'].value_counts().to_dict()}")
    print(f"\nAccuracy       : {accuracy:.4f}")
    print(f"Macro precision: {macro[0]:.4f}")
    print(f"Macro recall   : {macro[1]:.4f}")
    print(f"Macro F1       : {macro[2]:.4f}")

    print("\nPer-class metrics")
    print(f"  {'label':<10}{'precision':>11}{'recall':>9}{'f1':>9}{'support':>9}")
    for i, label in enumerate(LABEL_ORDER):
        print(
            f"  {label:<10}{per_class[0][i]:>11.4f}{per_class[1][i]:>9.4f}"
            f"{per_class[2][i]:>9.4f}{per_class[3][i]:>9}"
        )

    matrix = confusion_matrix(true, predicted, labels=LABEL_ORDER)
    print("\nConfusion matrix (rows = true, columns = predicted)")
    print(f"  {'':<10}" + "".join(f"{label:>10}" for label in LABEL_ORDER))
    for label, row in zip(LABEL_ORDER, matrix):
        print(f"  {label:<10}" + "".join(f"{value:>10}" for value in row))

    print("\nClassification report")
    print(
        classification_report(
            true, predicted, labels=LABEL_ORDER, zero_division=0, digits=4
        )
    )

    print(f"Total inference time      : {elapsed:.2f} s")
    print(f"Average time per example  : {elapsed / len(frame) * 1000:.1f} ms")


def report_neutral(frame: pd.DataFrame, elapsed: float) -> None:
    """Describe how the binary model behaves on neutral text.

    No accuracy is computed: there is no correct answer available to a model
    that cannot output NEUTRAL.
    """
    total = len(frame)
    counts = frame["predicted_label"].value_counts()
    confidence = frame["confidence"]

    print("\n" + "=" * 62)
    print("NEUTRAL DIAGNOSTIC (no accuracy - the model has no neutral class)")
    print("=" * 62)
    print(f"Neutral examples          : {total}")
    for label in LABEL_ORDER:
        count = int(counts.get(label, 0))
        print(f"  predicted {label:<9}: {count:>5}  ({count / total:.1%})")

    print(f"\nMean confidence           : {confidence.mean():.4f}")
    print(f"Median confidence         : {confidence.median():.4f}")
    print(f"Confidence >= 0.90        : {(confidence >= 0.90).mean():.1%}")
    print(f"Confidence >= 0.95        : {(confidence >= 0.95).mean():.1%}")

    print(f"\nTotal inference time      : {elapsed:.2f} s")
    print(f"Average time per example  : {elapsed / total * 1000:.1f} ms")


def print_methodology(binary_count: int, neutral_count: int) -> None:
    """State what these numbers do and do not mean."""
    print("\n" + "=" * 62)
    print("METHODOLOGY NOTE")
    print("=" * 62)
    print(
        f"- Binary metrics are calculated only on the {binary_count} "
        f"positive/negative\n  Financial PhraseBank examples.\n"
        f"- The {neutral_count} neutral examples are analyzed separately because "
        f"the\n  SST-2 DistilBERT checkpoint has no neutral class.\n"
        f"- Financial PhraseBank is financial-news/press-release language, while\n"
        f"  the checkpoint was fine-tuned on SST-2 movie-review sentiment, so this\n"
        f"  is an out-of-domain evaluation.\n"
        f"- The benchmark is sentence-level, while the Streamlit application\n"
        f"  performs article-level aggregation, so these metrics are not\n"
        f"  article-level accuracy."
    )


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def main() -> None:
    binary = load_csv(BINARY_CSV, ("text", "true_label"))
    check_labels(binary, set(LABEL_ORDER), BINARY_CSV)

    full = load_csv(FULL_CSV, ("text", "true_label"))
    neutral = full[full["true_label"] == "NEUTRAL"].reset_index(drop=True)
    if neutral.empty:
        raise ValueError(f"{FULL_CSV} contains no NEUTRAL rows.")
    check_labels(neutral, {"NEUTRAL"}, FULL_CSV)

    print(f"Model: {MODEL_CHECKPOINT}")
    tokenizer, model, device = load_sentiment_model()  # loaded once, not timed
    print(f"Device: {device}")

    # --- binary subset ---
    rows, binary_time = run_predictions(
        binary["text"].tolist(), tokenizer, model, device, "Binary subset"
    )
    binary_results = pd.concat([binary, pd.DataFrame(rows)], axis=1)
    binary_results["correct"] = (
        binary_results["true_label"] == binary_results["predicted_label"]
    )
    binary_results[
        [
            "text",
            "true_label",
            "predicted_label",
            "confidence",
            "positive_probability",
            "negative_probability",
            "correct",
        ]
    ].to_csv(BINARY_RESULTS_CSV, index=False)

    # --- neutral subset ---
    rows, neutral_time = run_predictions(
        neutral["text"].tolist(), tokenizer, model, device, "Neutral subset"
    )
    neutral_results = pd.concat([neutral, pd.DataFrame(rows)], axis=1)
    neutral_results[
        [
            "text",
            "predicted_label",
            "confidence",
            "positive_probability",
            "negative_probability",
        ]
    ].to_csv(NEUTRAL_RESULTS_CSV, index=False)

    report_binary(binary_results, binary_time)
    report_neutral(neutral_results, neutral_time)

    print(f"\nSaved: {BINARY_RESULTS_CSV}")
    print(f"Saved: {NEUTRAL_RESULTS_CSV}")

    print_methodology(len(binary_results), len(neutral_results))


if __name__ == "__main__":
    main()
