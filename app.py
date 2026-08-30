"""Streamlit interface for the Live News Sentiment Analyzer.

    streamlit run app.py

This file wires the existing modules together and renders results. No
scraping, cleaning, model or inference logic lives here.

Live predictions come only from the pretrained DistilBERT checkpoint. The
TF-IDF + Logistic Regression model is an offline evaluation baseline and is
deliberately NOT used here: it was trained on Financial PhraseBank financial
text, while this application accepts general news.
"""

import pandas as pd
import streamlit as st

from src.dl.inference import InferenceError, predict_sentiment
from src.dl.model import MODEL_CHECKPOINT, load_sentiment_model
from src.scraper.news_scraper import ScraperError, fetch_article
from src.scraper.text_cleaner import clean_text

PREVIEW_CHARS = 3000

st.set_page_config(page_title="News Sentiment Analyzer", page_icon="📰")


@st.cache_resource(show_spinner="Loading DistilBERT model...")
def get_model():
    """Load the model once per session instead of on every rerun."""
    return load_sentiment_model()


def analyze(url: str):
    """Scrape, clean and score one article. Returns (article, result)."""
    article = fetch_article(url)
    cleaned = clean_text(article.text)
    tokenizer, model, device = get_model()
    return article, predict_sentiment(cleaned, tokenizer, model, device)


def show_results(article, result) -> None:
    """Render the prediction for one article."""
    st.subheader(article.title)
    st.caption(article.url)

    left, right = st.columns(2)
    left.metric("Predicted sentiment", result.sentiment)
    right.metric("Model confidence", f"{result.confidence:.1%}")
    st.caption(
        "Model confidence is the averaged output probability of the predicted "
        "class. It is not a calibrated probability that the prediction is correct."
    )

    st.write(
        f"Positive probability: **{result.positive_probability:.1%}** &nbsp;·&nbsp; "
        f"Negative probability: **{result.negative_probability:.1%}**"
    )
    st.write(
        f"Chunks analyzed: **{result.num_chunks}** &nbsp;·&nbsp; "
        f"Total tokens: **{result.total_tokens}** &nbsp;·&nbsp; "
        f"Article length: **{len(article.text):,} characters**"
    )

    with st.expander("Article text preview"):
        preview = article.text[:PREVIEW_CHARS]
        if len(article.text) > PREVIEW_CHARS:
            preview += "\n\n[...]"
        st.text(preview)

    with st.expander(f"Chunk analysis ({result.num_chunks} chunks)"):
        st.caption(
            "Articles longer than DistilBERT's 512-token limit are split into "
            "overlapping 510-token chunks. Each chunk is scored separately and "
            "the article-level result is the mean of the chunk probabilities."
        )
        table = pd.DataFrame(result.chunk_predictions)
        st.dataframe(table, hide_index=True, use_container_width=True)

        if result.num_chunks > 1:
            labels = table["label"].nunique()
            if labels > 1:
                st.info(
                    "Chunks disagree with each other, so different sections of "
                    "this article carry different sentiment. The reported result "
                    "is the average across all chunks."
                )


def show_model_notes() -> None:
    """State plainly what the model is and what it cannot do."""
    with st.expander("About the model and its limitations"):
        st.markdown(f"""
**Model:** `{MODEL_CHECKPOINT}` — a pretrained DistilBERT checkpoint used
as-is for inference. It was not trained or fine-tuned as part of this project.

**Binary only.** The checkpoint predicts POSITIVE or NEGATIVE. There is
**no NEUTRAL class**, so factual or neutral reporting is still forced into one
of the two classes.

**Out of domain.** The checkpoint was fine-tuned on SST-2, a movie-review
sentiment dataset. News and financial writing differ in vocabulary, tone and
in what "positive" means, so predictions on news should be read as model
output rather than ground truth.

**Confidence is not accuracy.** The score shown is an averaged softmax
probability. In offline testing the model produced high-confidence
predictions on neutral sentences where no correct binary answer existed.

**No article-level accuracy claim.** This project's benchmarks are
sentence-level, measured on the Financial PhraseBank dataset. The
article-level chunking and aggregation used here have not been benchmarked,
so no accuracy figure applies to the predictions on this page.
""")


def main() -> None:
    st.title("📰 News Sentiment Analyzer")
    st.write(
        "Enter a news article URL. The article is scraped, cleaned and "
        "classified with a pretrained DistilBERT sentiment model."
    )

    with st.form("analyze_form"):
        url = st.text_input(
            "News article URL",
            placeholder="https://www.example.com/news/some-article",
        )
        submitted = st.form_submit_button("Analyze", type="primary")

    if submitted:
        if not url.strip():
            st.warning("Please enter a URL first.")
            st.session_state.pop("result", None)
        else:
            try:
                with st.spinner("Scraping and analyzing article..."):
                    st.session_state["result"] = analyze(url)
            except ScraperError as error:
                st.session_state.pop("result", None)
                st.error(f"Could not read the article: {error}")
            except InferenceError as error:
                st.session_state.pop("result", None)
                st.error(f"Could not analyze the text: {error}")
            except Exception as error:  # unexpected: show it, do not crash
                st.session_state.pop("result", None)
                st.error(f"Unexpected error: {type(error).__name__}: {error}")

    # Kept in session state so results survive reruns triggered by expanders.
    if "result" in st.session_state:
        show_results(*st.session_state["result"])

    st.divider()
    show_model_notes()
    st.caption(
        "Some news sites block automated requests; those URLs will return an "
        "error rather than a prediction."
    )


if __name__ == "__main__":
    main()
