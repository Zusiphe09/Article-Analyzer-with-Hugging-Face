import os
import requests
import streamlit as st
from transformers import pipeline


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Article Analyzer",
    layout="wide"
)


# ============================================================
# CONFIGURATION
# ============================================================

API_URL = "https://openrouter.ai/api/v1/chat/completions"

MODEL = "openrouter/free"


# ============================================================
# LOAD SENTIMENT MODEL
# ============================================================

@st.cache_resource
def load_sentiment_model():

    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )


sentiment_pipeline = load_sentiment_model()


# ============================================================
# GET OPENROUTER API KEY
# ============================================================

def get_api_key():

    try:

        api_key = st.secrets["OPENROUTER_API_KEY"]

        if api_key:
            return api_key

    except Exception:
        pass

    return os.environ.get("OPENROUTER_API_KEY")


# ============================================================
# OPENROUTER AI FUNCTION
# ============================================================

def ask_ai(prompt):

    api_key = get_api_key()

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
# LANGUAGE DETECTION
# ============================================================

def detect_language(article):

    if not article or not article.strip():

        return "Unknown"


    prompt = f"""
Detect the language of the following article.

Return ONLY the name of the language.
Do not provide an explanation.

Article:

{article[:3000]}
"""


    result = ask_ai(prompt)

    if result.startswith("Error"):

        return "Unknown"


    return result.strip()


# ============================================================
# MULTILINGUAL SENTIMENT ANALYSIS
# ============================================================

def analyze_sentiment(article):

    if not article or not article.strip():

        return {
            "error": "Please paste an article."
        }


    try:

        # Detect language first
        language = detect_language(article)


        # ----------------------------------------------------
        # English sentiment model
        # ----------------------------------------------------

        english_languages = [
            "english"
        ]


        if language.lower() in english_languages:

            text = article[:512]


            results = sentiment_pipeline(
                text,
                top_k=2,
                truncation=True
            )


            if isinstance(results[0], list):

                results = results[0]


            positive_score = 0.0
            negative_score = 0.0


            for result in results:

                label = result["label"].upper()

                score = result["score"] * 100


                if label == "POSITIVE":

                    positive_score = score


                elif label == "NEGATIVE":

                    negative_score = score


            if positive_score >= negative_score:

                sentiment = "POSITIVE"

                confidence = positive_score

            else:

                sentiment = "NEGATIVE"

                confidence = negative_score


        # ----------------------------------------------------
        # Multilingual AI sentiment analysis
        # ----------------------------------------------------

        else:

            prompt = f"""
Analyze the sentiment of the following article.

The article may be written in any language.

Return the following information:

Overall Sentiment: POSITIVE or NEGATIVE
Positive Score: a number between 0 and 100
Negative Score: a number between 0 and 100
Confidence Score: a number between 0 and 100

The positive and negative scores should add up to approximately
100.

Do not return NEUTRAL.

Return the result in this exact format:

Overall Sentiment: POSITIVE
Positive Score: 75.00
Negative Score: 25.00
Confidence Score: 75.00

Article:

{article[:6000]}
"""


            ai_result = ask_ai(prompt)


            sentiment = "POSITIVE"

            positive_score = 50.0

            negative_score = 50.0

            confidence = 50.0


            for line in ai_result.splitlines():

                line = line.strip()


                if line.lower().startswith(
                    "overall sentiment:"
                ):

                    value = line.split(
                        ":",
                        1
                    )[1].strip().upper()


                    if "NEGATIVE" in value:

                        sentiment = "NEGATIVE"

                    else:

                        sentiment = "POSITIVE"


                elif line.lower().startswith(
                    "positive score:"
                ):

                    try:

                        positive_score = float(
                            line.split(
                                ":",
                                1
                            )[1].strip()
                        )

                    except ValueError:

                        pass


                elif line.lower().startswith(
                    "negative score:"
                ):

                    try:

                        negative_score = float(
                            line.split(
                                ":",
                                1
                            )[1].strip()
                        )

                    except ValueError:

                        pass


                elif line.lower().startswith(
                    "confidence score:"
                ):

                    try:

                        confidence = float(
                            line.split(
                                ":",
                                1
                            )[1].strip()
                        )

                    except ValueError:

                        pass


            positive_score = max(
                0,
                min(100, positive_score)
            )


            negative_score = max(
                0,
                min(100, negative_score)
            )


            confidence = max(
                0,
                min(100, confidence)
            )


        return {
            "language": language,
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

The article may be written in any language.

Write the summary in the same language as the original article.

Requirements:

- Identify the main topic.
- Highlight the most important points.
- Mention important benefits or opportunities.
- Mention important challenges or risks.
- Do not invent information.
- Keep the summary easy to read.
- Use short paragraphs or bullet points where appropriate.
- Do not mention these instructions.

Article:

{article}
"""


    return ask_ai(prompt)


# ============================================================
# CITATION GENERATOR
# ============================================================

def generate_citation(
    title,
    author,
    publication,
    date,
    url,
    style
):

    title = title.strip()
    author = author.strip()
    publication = publication.strip()
    date = date.strip()
    url = url.strip()


    if not title:

        return "Please enter the article title."


    # --------------------------------------------------------
    # APA 7
    # --------------------------------------------------------

    if style == "APA 7":

        citation = (
            f"{author}. ({date}). "
            f"{title}. {publication}. {url}"
        )


    # --------------------------------------------------------
    # MLA 9
    # --------------------------------------------------------

    elif style == "MLA 9":

        citation = (
            f'{author}. "{title}." '
            f"{publication}, {date}, {url}."
        )


    # --------------------------------------------------------
    # Harvard
    # --------------------------------------------------------

    elif style == "Harvard":

        citation = (
            f"{author} ({date}) '{title}', "
            f"{publication}. Available at: {url}."
        )


    # --------------------------------------------------------
    # Chicago
    # --------------------------------------------------------

    elif style == "Chicago":

        citation = (
            f'{author}. "{title}." '
            f"{publication}, {date}. {url}."
        )


    else:

        citation = (
            f"{author}. {title}. "
            f"{publication}. {date}. {url}"
        )


    return citation


# ============================================================
# FORMAT SENTIMENT FOR DOWNLOAD
# ============================================================

def sentiment_download_text(result):

    return f"""
ARTICLE SENTIMENT ANALYSIS

Detected Language:
{result.get("language", "Unknown")}

Overall Sentiment:
{result["sentiment"]}

Confidence Score:
{result["confidence"]:.2f}%

Positive Score:
{result["positive"]:.2f}%

Negative Score:
{result["negative"]:.2f}%

"""


# ============================================================
# FORMAT FULL ANALYSIS FOR DOWNLOAD
# ============================================================

def full_analysis_download_text(
    sentiment_result,
    summary_result
):

    return f"""
ARTICLE ANALYSIS

========================================

SENTIMENT ANALYSIS

Detected Language:
{sentiment_result.get("language", "Unknown")}

Overall Sentiment:
{sentiment_result["sentiment"]}

Confidence Score:
{sentiment_result["confidence"]:.2f}%

Positive Score:
{sentiment_result["positive"]:.2f}%

Negative Score:
{sentiment_result["negative"]:.2f}%


========================================

ARTICLE SUMMARY

{summary_result}

"""


# ============================================================
# HEADER
# ============================================================

st.title("Article Analyzer")

st.markdown(
    "Paste an article below and choose an analysis option."
)


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Sentiment Analysis",
        "Summarize",
        "Full Analysis",
        "Citation Generator"
    ]
)


# ============================================================
# SENTIMENT ANALYSIS
# ============================================================

with tab1:

    st.subheader("Sentiment Analysis")


    sentiment_input = st.text_area(
        "Article",
        height=300,
        placeholder="Paste your article here...",
        key="sentiment_input"
    )


    if st.button(
        "Analyze Sentiment",
        key="sentiment_button"
    ):


        if not sentiment_input.strip():

            st.warning(
                "Please paste an article."
            )


        else:

            with st.spinner(
                "Analyzing sentiment..."
            ):

                result = analyze_sentiment(
                    sentiment_input
                )


            if "error" in result:

                st.error(
                    result["error"]
                )


            else:

                language = result["language"]

                sentiment = result["sentiment"]

                confidence = result["confidence"]

                positive = result["positive"]

                negative = result["negative"]


                st.caption(
                    f"Detected Language: {language}"
                )


                # Overall sentiment

                if sentiment == "POSITIVE":

                    st.success(
                        f"Overall Sentiment: {sentiment}"
                    )

                else:

                    st.error(
                        f"Overall Sentiment: {sentiment}"
                    )


                # Scores

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


                # Breakdown

                st.subheader(
                    "Sentiment Breakdown"
                )


                st.write(
                    f"Positive: {positive:.2f}%"
                )


                st.progress(
                    min(
                        positive / 100,
                        1.0
                    )
                )


                st.write(
                    f"Negative: {negative:.2f}%"
                )


                st.progress(
                    min(
                        negative / 100,
                        1.0
                    )
                )


                # Download

                download_text = sentiment_download_text(
                    result
                )


                st.download_button(
                    label="Download Sentiment Analysis",
                    data=download_text,
                    file_name="sentiment_analysis.txt",
                    mime="text/plain"
                )


# ============================================================
# SUMMARIZE
# ============================================================

with tab2:

    st.subheader("Article Summary")


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

            st.warning(
                "Please paste an article."
            )


        else:

            with st.spinner(
                "Generating summary..."
            ):

                summary_result = summarize_article(
                    summary_input
                )


            st.subheader("Summary")


            st.markdown(
                summary_result
            )


            # Download summary

            st.download_button(
                label="Download Summary",
                data=summary_result,
                file_name="article_summary.txt",
                mime="text/plain"
            )


# ============================================================
# FULL ANALYSIS
# ============================================================

with tab3:

    st.subheader("Full Article Analysis")


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

            st.warning(
                "Please paste an article."
            )


        else:

            # Sentiment

            with st.spinner(
                "Analyzing sentiment..."
            ):

                sentiment_result = analyze_sentiment(
                    full_input
                )


            # Summary

            with st.spinner(
                "Generating summary..."
            ):

                summary_result = summarize_article(
                    full_input
                )


            if "error" in sentiment_result:

                st.error(
                    sentiment_result["error"]
                )


            else:

                language = sentiment_result["language"]

                sentiment = sentiment_result["sentiment"]

                confidence = sentiment_result["confidence"]

                positive = sentiment_result["positive"]

                negative = sentiment_result["negative"]


                st.caption(
                    f"Detected Language: {language}"
                )


                # ------------------------------------------------
                # SENTIMENT
                # ------------------------------------------------

                st.divider()

                st.header(
                    "Sentiment Analysis"
                )


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
                    min(
                        positive / 100,
                        1.0
                    )
                )


                st.write(
                    f"Negative: {negative:.2f}%"
                )


                st.progress(
                    min(
                        negative / 100,
                        1.0
                    )
                )


                # ------------------------------------------------
                # SUMMARY
                # ------------------------------------------------

                st.divider()

                st.header(
                    "Article Summary"
                )


                st.markdown(
                    summary_result
                )


                # ------------------------------------------------
                # DOWNLOAD FULL ANALYSIS
                # ------------------------------------------------

                full_download = (
                    full_analysis_download_text(
                        sentiment_result,
                        summary_result
                    )
                )


                st.download_button(
                    label="Download Full Analysis",
                    data=full_download,
                    file_name="full_article_analysis.txt",
                    mime="text/plain"
                )


# ============================================================
# CITATION GENERATOR
# ============================================================

with tab4:

    st.subheader("Citation Generator")


    title = st.text_input(
        "Article Title",
        placeholder="Enter the article title"
    )


    author = st.text_input(
        "Author",
        placeholder="Enter the author name"
    )


    publication = st.text_input(
        "Website or Publication",
        placeholder="Enter the website or publication"
    )


    date = st.text_input(
        "Publication Date",
        placeholder="Example: 10 August 2026"
    )


    url = st.text_input(
        "Article URL",
        placeholder="Paste the article URL"
    )


    style = st.selectbox(
        "Citation Style",
        [
            "APA 7",
            "MLA 9",
            "Harvard",
            "Chicago"
        ]
    )


    if st.button(
        "Generate Citation",
        key="citation_button"
    ):


        if not title:

            st.warning(
                "Please enter the article title."
            )


        else:

            citation = generate_citation(
                title,
                author,
                publication,
                date,
                url,
                style
            )


            st.subheader(
                "Generated Citation"
            )


            st.text_area(
                "Citation",
                value=citation,
                height=150
            )


            st.download_button(
                label="Download Citation",
                data=citation,
                file_name="citation.txt",
                mime="text/plain"
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Article Analyzer | Hugging Face + OpenRouter + Streamlit"
)
