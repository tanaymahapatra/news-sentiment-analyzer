"""Download and prepare Financial PhraseBank for evaluation.

Downloads the dataset once, maps its integer labels to explicit label names,
and writes two CSV files:

    data/evaluation/financial_phrasebank_full.csv    all 4,846 sentences
    data/evaluation/financial_phrasebank_binary.csv  positive/negative only

No model, no inference, no training here. Run once:

    python -m src.evaluation.prepare_dataset
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Parquet mirror of Financial PhraseBank (Malo et al., 2014), config
# `sentences_50agree`. The original `takala/financial_phrasebank` repo is
# script-based and no longer loads with datasets>=3.
HF_DATASET = "warwickai/financial_phrasebank_mirror"

# Financial PhraseBank's integer labels. NOTE: this ordering is NOT the same
# as the model's id2label (where NEGATIVE=0, POSITIVE=1), so labels are always
# mapped by name and never by index.
FPB_ID_TO_LABEL = {
    0: "NEGATIVE",
    1: "NEUTRAL",
    2: "POSITIVE",
}

# Published class counts for sentences_50agree. Used as an integrity check
# that the mirror still matches the original dataset.
EXPECTED_COUNTS = {"NEGATIVE": 604, "NEUTRAL": 2879, "POSITIVE": 1363}

# The model can only predict these two classes.
BINARY_LABELS = ("NEGATIVE", "POSITIVE")

OUTPUT_DIR = Path("data/evaluation")
FULL_CSV = OUTPUT_DIR / "financial_phrasebank_full.csv"
BINARY_CSV = OUTPUT_DIR / "financial_phrasebank_binary.csv"


def download_dataset() -> pd.DataFrame:
    """Download the dataset and return it as a DataFrame with mapped labels."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit(
            "The 'datasets' library is required.\n"
            "Install it with:  pip install datasets"
        )

    split = load_dataset(HF_DATASET)["train"]
    frame = pd.DataFrame({"text": split["sentence"], "label_id": split["label"]})

    unknown = set(frame["label_id"]) - set(FPB_ID_TO_LABEL)
    if unknown:
        raise ValueError(f"Dataset contains unexpected label ids: {unknown}")

    frame["true_label"] = frame["label_id"].map(FPB_ID_TO_LABEL)
    return frame[["text", "true_label"]]


def verify_counts(frame: pd.DataFrame) -> None:
    """Warn if the mirror no longer matches the published class counts."""
    counts = frame["true_label"].value_counts().to_dict()
    if counts != EXPECTED_COUNTS:
        print("WARNING: class counts differ from the published dataset.")
        print(f"  expected: {EXPECTED_COUNTS}")
        print(f"  found   : {counts}")
        print("  Check the label mapping before trusting any metrics.\n")


def print_distribution(frame: pd.DataFrame, name: str) -> None:
    """Print class counts and percentages for one DataFrame."""
    total = len(frame)
    print(f"{name} ({total} examples)")
    for label, count in frame["true_label"].value_counts().items():
        print(f"  {label:<9} {count:>5}  ({count / total:.1%})")
    print()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {HF_DATASET} ...\n")
    full = download_dataset()
    verify_counts(full)

    binary = full[full["true_label"].isin(BINARY_LABELS)].reset_index(drop=True)

    full.to_csv(FULL_CSV, index=False)
    binary.to_csv(BINARY_CSV, index=False)

    print_distribution(full, "FULL DATASET")
    print_distribution(binary, "BINARY SUBSET (used for accuracy/F1)")

    neutral = len(full) - len(binary)
    print(
        f"Neutral examples excluded from binary metrics: {neutral} "
        f"({neutral / len(full):.1%} of the dataset)"
    )
    print("These are kept in the full file for the separate neutral diagnostic.\n")

    print(f"Saved: {FULL_CSV}")
    print(f"Saved: {BINARY_CSV}")


if __name__ == "__main__":
    main()
