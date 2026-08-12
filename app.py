import os
import requests
import streamlit as st
from transformers import pipeline


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Article Analyzer",
    page_icon="📰",
    layout="wide"
)


# ============================================================
# CONFIGURATION
# ============================================================

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

API_URL = "https://openrouter.ai/api/v1/chat/completions"

MODEL = "openrouter/free"


# ============================================================
# HUGGING FACE SENTIMENT ANALYSIS MODEL
# ============================================================

@st.cache_resource
def load_sentiment_model():

    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )


sentiment_pipeline = load_sentiment_model()


# ============================================================
# OPENROUTER AI FUNCTION
# ============================================================

def ask_ai(prompt):

    api_key = os.environ.get("OPENROUTER_API_KEY")

    if not api_key:
        return "Error: OPENROUTER_API_KEY is not set."

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }

    try:

        response = requests.post(
            API_URL,
            headers=headers,
            json=data,
            timeout=60
        )

        if response.status_code != 200:

            return (
                f"OpenRouter Error: {response.status_code}\n\n"
                f"{response.text}"
            )

        result = response.json()

        return result["choices"][0]["message"]["content"]

    except requests.exceptions.RequestException as e:

        return f"Request Error: {str(e)}"

    except Exception as e:

        return f"Error: {str(e)}"


# ============================================================
# SENTIMENT ANALYSIS
# ============================================================

def analyze_sentiment(article):

    if not article or not article.strip():

        return {
            "error": "Please paste an article."
        }

    try:

        # Analyze first 512 characters
        text = article[:512]

        results = sentiment_pipeline(
            text,
            top_k=2,
            truncation=True
        )

        # Handle different Transformers output formats
        if isinstance(results[0], list):
            results = results[0]

        positive_score = 0.0
        negative_score = 0.0

        # Extract sentiment scores
        for result in results:

            label = result["label"].upper()

            score = result["score"] * 100

            if label == "POSITIVE":

                positive_score = score

            elif label == "NEGATIVE":

                negative_score = score

        # Determine overall sentiment
        if positive_score >= negative_score:

            sentiment = "POSITIVE"

            confidence = positive_score

        else:

            sentiment = "NEGATIVE"

            confidence = negative_score

        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "positive": positive_score,
            "negative": negative_score
        }

    except Exception as e:

        return {
            "error": str(e)
        }


# ============================================================
# ARTICLE SUMMARIZATION
# ============================================================

def summarize_article(article):

    if not article or not article.strip():

        return "Please paste an article."

    prompt = f"""
You are an AI article summarization assistant.

Summarize the following article clearly and concisely.

Requirements:

- Identify the main topic.
- Highlight the most important points.
- Mention important benefits or opportunities.
- Mention important challenges or risks.
- Do not invent information.
- Keep the summary easy to read.

Article:

{article}
"""

    return ask_ai(prompt)


# ============================================================
# FULL ANALYSIS
# ============================================================

def full_analysis(article):

    sentiment_result = analyze_sentiment(article)

    if "error" in sentiment_result:

        return (
            sentiment_result["error"],
            ""
        )

    summary_result = summarize_article(article)

    return (
        sentiment_result,
        summary_result
    )


# ============================================================
# HEADER
# ============================================================

st.title("📰 Article Analyzer")

st.markdown(
    """
Analyze articles using **AI-powered sentiment analysis**
and **article summarization**.

Paste an article below and choose an analysis option.
"""
)


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3 = st.tabs(
    [
        "📊 Sentiment Analysis",
        "📝 Summarize",
        "🔍 Full Analysis"
    ]
)


# ============================================================
# SENTIMENT ANALYSIS TAB
# ============================================================

with tab1:

    st.subheader("Sentiment Analysis")

    st.write(
        """
        Analyze the sentiment of an article using a
        pre-trained Hugging Face model.
        
        The model provides:
        - Positive sentiment percentage
        - Negative sentiment percentage
        - Overall sentiment
        - Confidence score
        """
    )

    sentiment_input = st.text_area(
        "Article",
        height=300,
        placeholder="Paste your article here..."
    )

    if st.button(
        "Analyze Sentiment",
        key="sentiment_button"
    ):

        if not sentiment_input.strip():

            st.warning("Please paste an article.")

        else:

            with st.spinner("Analyzing sentiment..."):

                result = analyze_sentiment(
                    sentiment_input
                )

            if "error" in result:

                st.error(result["error"])

            else:

                sentiment = result["sentiment"]

                confidence = result["confidence"]

                positive = result["positive"]

                negative = result["negative"]


                # Overall sentiment
                st.subheader("Overall Sentiment")

                if sentiment == "POSITIVE":

                    st.success(
                        f"🟢 {sentiment}"
                    )

                else:

                    st.error(
                        f"🔴 {sentiment}"
                    )


                # Confidence
                st.subheader("Confidence Score")

                st.metric(
                    "Confidence",
                    f"{confidence:.2f}%"
                )


                # Sentiment scores
                st.subheader("Sentiment Scores")

                col1, col2 = st.columns(2)

                with col1:

                    st.metric(
                        "Positive",
                        f"{positive:.2f}%"
                    )

                    st.progress(
                        min(positive / 100, 1.0)
                    )

                with col2:

                    st.metric(
                        "Negative",
                        f"{negative:.2f}%"
                    )

                    st.progress(
                        min(negative / 100, 1.0)
                    )


                # Explanation
                st.info(
                    f"""
                    The model is **{confidence:.2f}% confident**
                    that this article has a
                    **{sentiment.lower()}** sentiment.
                    """
                )


                # Model information
                st.caption(
                    "Model: "
                    "distilbert-base-uncased-finetuned-sst-2-english"
                )


# ============================================================
# SUMMARIZATION TAB
# ============================================================

with tab2:

    st.subheader("Article Summarization")

    st.write(
        """
        Generate a clear and concise summary of an article
        using OpenRouter AI.
        """
    )

    summary_input = st.text_area(
        "Article",
        height=300,
        placeholder="Paste your article here...",
        key="summary_input"
    )

    if st.button(
        "Generate Summary",
        key="summary_button"
    ):

        if not summary_input.strip():

            st.warning("Please paste an article.")

        else:

            with st.spinner(
                "Generating article summary..."
            ):

                summary_result = summarize_article(
                    summary_input
                )

            st.subheader("Article Summary")

            st.markdown(summary_result)


# ============================================================
# FULL ANALYSIS TAB
# ============================================================

with tab3:

    st.subheader("Full Article Analysis")

    st.write(
        """
        Perform both sentiment analysis and AI-powered
        article summarization.
        """
    )

    full_input = st.text_area(
        "Article",
        height=300,
        placeholder="Paste your article here...",
        key="full_input"
    )

    if st.button(
        "Analyze Article",
        key="full_button"
    ):

        if not full_input.strip():

            st.warning("Please paste an article.")

        else:

            with st.spinner(
                "Performing full article analysis..."
            ):

                sentiment_result = analyze_sentiment(
                    full_input
                )

                summary_result = summarize_article(
                    full_input
                )


            # ------------------------------------------------
            # SENTIMENT RESULTS
            # ------------------------------------------------

            st.divider()

            st.header("📊 Sentiment Analysis")

            if "error" in sentiment_result:

                st.error(
                    sentiment_result["error"]
                )

            else:

                sentiment = sentiment_result["sentiment"]

                confidence = sentiment_result["confidence"]

                positive = sentiment_result["positive"]

                negative = sentiment_result["negative"]


                if sentiment == "POSITIVE":

                    st.success(
                        f"Overall Sentiment: {sentiment}"
                    )

                else:

                    st.error(
                        f"Overall Sentiment: {sentiment}"
                    )


                col1, col2, col3 = st.columns(3)

                with col1:

                    st.metric(
                        "Confidence",
                        f"{confidence:.2f}%"
                    )

                with col2:

                    st.metric(
                        "Positive",
                        f"{positive:.2f}%"
                    )

                with col3:

                    st.metric(
                        "Negative",
                        f"{negative:.2f}%"
                    )


                st.subheader(
                    "Sentiment Breakdown"
                )

                st.write(
                    f"Positive: {positive:.2f}%"
                )

                st.progress(
                    min(positive / 100, 1.0)
                )

                st.write(
                    f"Negative: {negative:.2f}%"
                )

                st.progress(
                    min(negative / 100, 1.0)
                )


            # ------------------------------------------------
            # SUMMARY
            # ------------------------------------------------

            st.divider()

            st.header("📝 Article Summary")

            st.markdown(summary_result)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Article Analyzer | "
    "Hugging Face + OpenRouter + Streamlit"
)
