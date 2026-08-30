"""Compare TF-IDF + Logistic Regression against pretrained DistilBERT.

Both models are scored on exactly the same 394 held-out rows written by
src/ml/baseline.py. Nothing is trained or tuned here: the ML predictions are
read from disk, and DistilBERT runs zero-shot through the same path the
Streamlit app uses.

    python -m src.evaluation.compare_models
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from src.dl.inference import predict_sentiment
from src.dl.model import MODEL_CHECKPOINT, load_sentiment_model
from src.scraper.text_cleaner import clean_text

DATA_DIR = Path("data/evaluation")
TEST_SPLIT_CSV = DATA_DIR / "test_split.csv"
ML_RESULTS_CSV = DATA_DIR / "ml_results.csv"
COMPARISON_CSV = DATA_DIR / "model_comparison.csv"

LABEL_ORDER = ["NEGATIVE", "POSITIVE"]
PROGRESS_EVERY = 100


# --------------------------------------------------------------------------
# Loading and validation
# --------------------------------------------------------------------------


def load_csv(path: Path, required: tuple) -> pd.DataFrame:
    """Load a CSV and check required columns, uniqueness and labels."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run:  python -m src.ml.baseline")

    frame = pd.read_csv(path)

    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing required column(s): {missing}")
    if frame.empty:
        raise ValueError(f"{path} contains no rows.")
    if not frame["row_id"].is_unique:
        duplicates = frame.loc[frame["row_id"].duplicated(), "row_id"].tolist()
        raise ValueError(f"{path} has duplicate row_id values: {duplicates}")

    unexpected = set(frame["true_label"].unique()) - set(LABEL_ORDER)
    if unexpected:
        raise ValueError(f"{path} contains unexpected labels: {unexpected}")

    return frame


def load_aligned_data() -> pd.DataFrame:
    """Load both files and fail loudly on any mismatch between them."""
    test = load_csv(TEST_SPLIT_CSV, ("row_id", "text", "true_label"))
    ml = load_csv(ML_RESULTS_CSV, ("row_id", "text", "true_label", "predicted_label"))

    test_ids, ml_ids = set(test["row_id"]), set(ml["row_id"])
    if test_ids != ml_ids:
        raise ValueError(
            "row_id sets differ between the two files. No rows were dropped; "
            "re-run src.ml.baseline so both files come from one split.\n"
            f"  only in {TEST_SPLIT_CSV.name}: {sorted(test_ids - ml_ids)[:10]}\n"
            f"  only in {ML_RESULTS_CSV.name}: {sorted(ml_ids - test_ids)[:10]}"
        )

    unexpected = set(ml["predicted_label"].unique()) - set(LABEL_ORDER)
    if unexpected:
        raise ValueError(f"{ML_RESULTS_CSV} has unexpected predictions: {unexpected}")

    merged = test.merge(ml, on="row_id", suffixes=("", "_ml"), validate="one_to_one")

    mismatched = merged[merged["true_label"] != merged["true_label_ml"]]
    if not mismatched.empty:
        raise ValueError(
            f"true_label disagrees for row_id(s) {mismatched['row_id'].tolist()[:10]}. "
            "The two files describe different data."
        )

    mismatched_text = merged[merged["text"] != merged["text_ml"]]
    if not mismatched_text.empty:
        raise ValueError(
            f"text disagrees for row_id(s) {mismatched_text['row_id'].tolist()[:10]}. "
            "The ML results were produced from a different split."
        )

    merged = merged.rename(columns={"predicted_label": "ml_prediction"})
    keep = ["row_id", "text", "true_label", "ml_prediction"]
    for column in ("positive_probability", "negative_probability"):
        if column in merged.columns:
            merged = merged.rename(columns={column: f"ml_{column}"})
            keep.append(f"ml_{column}")

    return merged[keep]


# --------------------------------------------------------------------------
# DistilBERT predictions
# --------------------------------------------------------------------------


def run_distilbert(texts: List[str], tokenizer, model, device):
    """Score every row with the app's pipeline. Returns (rows, seconds)."""
    print(f"\nRunning DistilBERT on {len(texts)} held-out examples")
    rows = []

    start = time.perf_counter()
    for index, text in enumerate(texts, start=1):
        try:
            result = predict_sentiment(clean_text(text), tokenizer, model, device)
        except Exception as error:
            raise RuntimeError(
                f"Prediction failed on row {index - 1}: "
                f"{type(error).__name__}: {error}"
            ) from error

        rows.append(
            {
                "dl_prediction": result.sentiment,
                "dl_confidence": result.confidence,
                "dl_positive_probability": result.positive_probability,
                "dl_negative_probability": result.negative_probability,
            }
        )

        if index % PROGRESS_EVERY == 0 or index == len(texts):
            print(f"  {index}/{len(texts)}")

    return rows, time.perf_counter() - start


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def compute_metrics(true: List[str], predicted: List[str]) -> Dict[str, float]:
    """Accuracy plus macro and per-class precision/recall/F1."""
    macro = precision_recall_fscore_support(
        true, predicted, labels=LABEL_ORDER, average="macro", zero_division=0
    )
    per_class = precision_recall_fscore_support(
        true, predicted, labels=LABEL_ORDER, average=None, zero_division=0
    )

    metrics = {
        "Accuracy": accuracy_score(true, predicted),
        "Macro Precision": macro[0],
        "Macro Recall": macro[1],
        "Macro F1": macro[2],
    }
    for i, label in enumerate(LABEL_ORDER):
        name = label.capitalize()
        metrics[f"{name} Precision"] = per_class[0][i]
        metrics[f"{name} Recall"] = per_class[1][i]
        metrics[f"{name} F1"] = per_class[2][i]

    return metrics


def print_comparison(ml: Dict[str, float], dl: Dict[str, float]) -> None:
    """Side-by-side metric table with the difference."""
    print("\n" + "=" * 62)
    print("MODEL COMPARISON - IDENTICAL HELD-OUT TEST SET")
    print("=" * 62)
    print(f"{'Metric':<22}{'TF-IDF + LR':>14}{'DistilBERT':>14}{'Diff':>12}")
    print("-" * 62)
    for name in ml:
        difference = ml[name] - dl[name]
        print(f"{name:<22}{ml[name]:>14.4f}{dl[name]:>14.4f}{difference:>+12.4f}")
    print("\nDiff is TF-IDF + LR minus DistilBERT.")


def print_confusion(true: List[str], predicted: List[str], name: str) -> None:
    matrix = confusion_matrix(true, predicted, labels=LABEL_ORDER)
    print(f"\n{name} confusion matrix (rows = true, columns = predicted)")
    print(f"  {'':<10}" + "".join(f"{label:>10}" for label in LABEL_ORDER))
    for label, row in zip(LABEL_ORDER, matrix):
        print(f"  {label:<10}" + "".join(f"{value:>10}" for value in row))


def print_disagreement(frame: pd.DataFrame) -> None:
    """Where the two models agree, differ, and which one is right."""
    total = len(frame)
    both_right = int((frame["ml_correct"] & frame["dl_correct"]).sum())
    both_wrong = int((~frame["ml_correct"] & ~frame["dl_correct"]).sum())
    ml_only = int((frame["ml_correct"] & ~frame["dl_correct"]).sum())
    dl_only = int((~frame["ml_correct"] & frame["dl_correct"]).sum())
    disagree = int((~frame["models_agree"]).sum())

    print("\n" + "=" * 62)
    print("DISAGREEMENT ANALYSIS")
    print("=" * 62)
    for label, count in [
        ("Both correct", both_right),
        ("Both wrong", both_wrong),
        ("ML correct, DistilBERT wrong", ml_only),
        ("DistilBERT correct, ML wrong", dl_only),
    ]:
        print(f"  {label:<32}{count:>5}  ({count / total:.1%})")
    print(f"  {'Predictions that differ':<32}{disagree:>5}  ({disagree / total:.1%})")
    print(f"\n  Total examples: {total}")


def print_interpretation(ml: Dict[str, float], dl: Dict[str, float]) -> None:
    """Interpretation derived only from the measured numbers above."""
    print("\n" + "=" * 62)
    print("INTERPRETATION")
    print("=" * 62)

    gap = ml["Accuracy"] - dl["Accuracy"]
    if gap > 0:
        leader, trailer = "TF-IDF + Logistic Regression", "DistilBERT"
    elif gap < 0:
        leader, trailer = "DistilBERT", "TF-IDF + Logistic Regression"
    else:
        leader = trailer = None

    if leader is None:
        print("- Both models reached the same accuracy on this test set.")
    else:
        print(
            f"- {leader} scored higher than {trailer} on this test set "
            f"(accuracy difference {abs(gap):.4f})."
        )

    print(
        "- This is a legitimate result either way, but it is not a general\n"
        "  claim about classical ML versus transformers. The Logistic\n"
        "  Regression model was TRAINED on 1,573 in-domain Financial PhraseBank\n"
        "  sentences. The DistilBERT checkpoint was fine-tuned on SST-2\n"
        "  movie-review sentiment and is evaluated here zero-shot and\n"
        "  out-of-domain, having never seen financial text.\n"
        "- A fine-tuned DistilBERT on the same training split would be the\n"
        "  like-for-like transformer comparison. That was deliberately not done.\n"
        "- Financial PhraseBank labels reflect an investor's perspective on\n"
        "  expected business impact, while SST-2 labels reflect the writer's\n"
        "  expressed opinion. These are related but different tasks.\n"
        "- This comparison is SENTENCE-LEVEL. It does not validate the\n"
        "  overlapping-chunk splitting or the mean-probability aggregation the\n"
        "  Streamlit app uses for full articles, so these numbers are not\n"
        "  article-level accuracy.\n"
        "- The test set is 394 examples (121 negative), so small differences\n"
        "  between the two models are within sampling noise."
    )


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def main() -> None:
    data = load_aligned_data()
    print(f"Loaded {len(data)} held-out examples from {TEST_SPLIT_CSV}")
    print(f"Class distribution: {data['true_label'].value_counts().to_dict()}")

    print(f"\nModel: {MODEL_CHECKPOINT}")
    tokenizer, model, device = load_sentiment_model()  # loaded once, not timed
    print(f"Device: {device}")

    rows, elapsed = run_distilbert(data["text"].tolist(), tokenizer, model, device)
    data = pd.concat([data, pd.DataFrame(rows)], axis=1)

    data["ml_correct"] = data["true_label"] == data["ml_prediction"]
    data["dl_correct"] = data["true_label"] == data["dl_prediction"]
    data["models_agree"] = data["ml_prediction"] == data["dl_prediction"]

    true = data["true_label"].tolist()
    ml_metrics = compute_metrics(true, data["ml_prediction"].tolist())
    dl_metrics = compute_metrics(true, data["dl_prediction"].tolist())

    print_comparison(ml_metrics, dl_metrics)
    print_confusion(true, data["ml_prediction"].tolist(), "TF-IDF + LR")
    print_confusion(true, data["dl_prediction"].tolist(), "DistilBERT")
    print_disagreement(data)

    print("\n" + "=" * 62)
    print("DISTILBERT INFERENCE TIMING")
    print("=" * 62)
    print(f"  Total inference time     : {elapsed:.2f} s")
    print(f"  Average time per example : {elapsed / len(data) * 1000:.1f} ms")
    print("  (Model loading excluded. Not comparable to ML training time.)")

    data.to_csv(COMPARISON_CSV, index=False)
    print(f"\nSaved: {COMPARISON_CSV}")

    print_interpretation(ml_metrics, dl_metrics)


if __name__ == "__main__":
    main()

