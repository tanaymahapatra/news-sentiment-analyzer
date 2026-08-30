"""Classical baseline: TF-IDF + Logistic Regression on Financial PhraseBank.

Deliberately untuned. No grid search, no cross-validation, no threshold
tuning, no resampling. The point is a defensible reference number, not the
best possible classical model.

    python -m src.ml.baseline

Writes the held-out split to data/evaluation/test_split.csv so that
compare_models.py can score DistilBERT on exactly the same rows.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.scraper.text_cleaner import clean_text

DATA_DIR = Path("data/evaluation")
BINARY_CSV = DATA_DIR / "financial_phrasebank_binary.csv"
TEST_SPLIT_CSV = DATA_DIR / "test_split.csv"
ML_RESULTS_CSV = DATA_DIR / "ml_results.csv"

LABEL_ORDER = ["NEGATIVE", "POSITIVE"]
TEST_SIZE = 0.2
RANDOM_STATE = 42
TOP_FEATURES = 10


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------


def load_dataset() -> pd.DataFrame:
    """Load the binary subset, validate it, and attach a stable row_id."""
    if not BINARY_CSV.exists():
        raise FileNotFoundError(
            f"{BINARY_CSV} not found. Run:  python -m src.evaluation.prepare_dataset"
        )

    frame = pd.read_csv(BINARY_CSV)

    missing = [c for c in ("text", "true_label") if c not in frame.columns]
    if missing:
        raise ValueError(f"{BINARY_CSV} is missing required column(s): {missing}")
    if frame.empty:
        raise ValueError(f"{BINARY_CSV} contains no rows.")

    found = set(frame["true_label"].unique())
    if found != set(LABEL_ORDER):
        raise ValueError(
            f"{BINARY_CSV} must contain only {set(LABEL_ORDER)}, found {found}."
        )

    # row_id is the original position in the CSV and is the key that
    # compare_models.py will use to score the identical rows.
    frame = frame.reset_index(drop=True)
    frame["row_id"] = frame.index

    # clean_text is row-wise and stateless, so applying it before the split
    # cannot leak information between train and test.
    frame["clean_text"] = frame["text"].apply(clean_text)

    return frame


def split_dataset(frame: pd.DataFrame):
    """One stratified split, with the disjointness checks made explicit."""
    train, test = train_test_split(
        frame,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=frame["true_label"],
    )

    train_ids, test_ids = set(train["row_id"]), set(test["row_id"])
    assert not (train_ids & test_ids), "Leakage: row_id present in train and test."
    assert train_ids | test_ids == set(frame["row_id"]), "Split lost rows."
    assert len(train) + len(test) == len(frame), "Split changed the row count."

    return train.reset_index(drop=True), test.reset_index(drop=True)


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------


def build_pipeline() -> Pipeline:
    """TF-IDF then Logistic Regression, in one Pipeline.

    Keeping the vectorizer inside the Pipeline means it is fit during
    pipeline.fit(train) only, so the test vocabulary and IDF weights can
    never influence training.
    """
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                    stop_words=None,  # sklearn's list contains negations
                    max_features=None,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    solver="liblinear",
                    class_weight="balanced",
                    max_iter=1000,
                    C=1.0,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def show_top_features(pipeline: Pipeline) -> None:
    """Print the strongest features for each class. Diagnostic only."""
    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["clf"]

    # Binary Logistic Regression has one coefficient vector; positive values
    # push towards classes_[1]. Verified rather than assumed.
    assert list(classifier.classes_) == LABEL_ORDER, (
        f"Unexpected class order {classifier.classes_}; "
        "the feature signs below would be inverted."
    )

    names = vectorizer.get_feature_names_out()
    coefficients = classifier.coef_[0]
    ranked = coefficients.argsort()

    print(f"\nTop {TOP_FEATURES} features per class (diagnostic only)")
    print(f"  {'POSITIVE':<28}{'NEGATIVE'}")
    top_positive = ranked[::-1][:TOP_FEATURES]
    top_negative = ranked[:TOP_FEATURES]
    for pos, neg in zip(top_positive, top_negative):
        left = f"{names[pos]} ({coefficients[pos]:+.3f})"
        right = f"{names[neg]} ({coefficients[neg]:+.3f})"
        print(f"  {left:<28}{right}")


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def print_distribution(frame: pd.DataFrame, name: str) -> None:
    counts = frame["true_label"].value_counts().to_dict()
    print(f"  {name:<6} {len(frame):>5} examples  {counts}")


def report(test: pd.DataFrame, predicted, fit_time: float, predict_time: float) -> None:
    """Print the held-out test metrics."""
    true = test["true_label"].tolist()

    accuracy = accuracy_score(true, predicted)
    macro = precision_recall_fscore_support(
        true, predicted, labels=LABEL_ORDER, average="macro", zero_division=0
    )
    per_class = precision_recall_fscore_support(
        true, predicted, labels=LABEL_ORDER, average=None, zero_division=0
    )

    print("\n" + "=" * 62)
    print("TF-IDF + LOGISTIC REGRESSION - HELD-OUT TEST SET")
    print("=" * 62)
    print(f"Examples evaluated : {len(test)}")
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

    print(f"Training time            : {fit_time:.3f} s")
    print(f"Total prediction time    : {predict_time:.3f} s")
    print(f"Average time per example : {predict_time / len(test) * 1000:.3f} ms")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def main() -> None:
    frame = load_dataset()
    train, test = split_dataset(frame)

    print("Split (stratified, random_state=42)")
    print_distribution(train, "train")
    print_distribution(test, "test")

    pipeline = build_pipeline()

    start = time.perf_counter()
    pipeline.fit(train["clean_text"], train["true_label"])  # training rows only
    fit_time = time.perf_counter() - start

    start = time.perf_counter()
    predicted = pipeline.predict(test["clean_text"])
    predict_time = time.perf_counter() - start

    probabilities = pipeline.predict_proba(test["clean_text"])
    classes = list(pipeline.named_steps["clf"].classes_)

    report(test, predicted, fit_time, predict_time)
    show_top_features(pipeline)

    # Original (uncleaned) text is saved so compare_models.py runs the same
    # clean_text step the Streamlit app uses.
    test[["row_id", "text", "true_label"]].to_csv(TEST_SPLIT_CSV, index=False)

    results = test[["row_id", "text", "true_label"]].copy()
    results["predicted_label"] = predicted
    results["positive_probability"] = probabilities[:, classes.index("POSITIVE")].round(
        4
    )
    results["negative_probability"] = probabilities[:, classes.index("NEGATIVE")].round(
        4
    )
    results["correct"] = results["true_label"] == results["predicted_label"]
    results.to_csv(ML_RESULTS_CSV, index=False)

    print(f"\nSaved: {TEST_SPLIT_CSV}")
    print(f"Saved: {ML_RESULTS_CSV}")
    print(
        "\nUntuned baseline: no grid search, no cross-validation, no threshold\n"
        "tuning. Vectorizer fit inside the Pipeline on training rows only."
    )


if __name__ == "__main__":
    main()
