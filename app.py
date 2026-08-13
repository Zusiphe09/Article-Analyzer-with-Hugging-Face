import os
import requests
import streamlit as st
from transformers import pipeline
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_LEFT
from io import BytesIO


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Article Analyzer",
    page_icon=None,
    layout="wide"
)


# ============================================================
# CONFIGURATION
# ============================================================

API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "openrouter/free"


# ============================================================
# GET OPENROUTER API KEY
# ============================================================

def get_api_key():

    try:
        if "OPENROUTER_API_KEY" in st.secrets:
            return st.secrets["OPENROUTER_API_KEY"]
    except Exception:
        pass

    return os.environ.get("OPENROUTER_API_KEY")


# ============================================================
# HUGGING FACE SENTIMENT MODEL
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
            timeout=120
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
            "sentiment": "No article provided",
            "confidence": 0,
            "positive": 0,
            "negative": 0
        }

    try:

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

        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "positive": positive_score,
            "negative": negative_score
        }

    except Exception as e:

        return {
            "sentiment": f"Error: {str(e)}",
            "confidence": 0,
            "positive": 0,
            "negative": 0
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

Write the summary in the same language as the original article.

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
# ARTICLE PARAPHRASER
# ============================================================

def paraphrase_article(article):

    if not article or not article.strip():

        return "Please paste text to paraphrase."

    prompt = f"""
You are an AI paraphrasing assistant.

Rewrite the following text while preserving its original meaning.

Rules:

- Keep the original meaning.
- Do not add information.
- Do not remove important information.
- Use natural and clear language.
- Improve sentence structure where appropriate.
- Keep the same language as the original text.

Text:

{article}
"""

    return ask_ai(prompt)


# ============================================================
# CITATION GENERATOR
# ============================================================

def generate_citation(title, author, year, url, style):

    if not title.strip():

        return "Please enter the article title."

    prompt = f"""
Generate an academic citation using the information below.

Citation style:
{style}

Article title:
{title}

Author:
{author}

Year:
{year}

URL:
{url}

Instructions:

- Generate only the citation.
- Do not invent missing information.
- Follow the requested citation style.
"""

    return ask_ai(prompt)


# ============================================================
# FULL ANALYSIS
# ============================================================

def full_analysis(article):

    if not article or not article.strip():

        return None, "Please paste an article."

    sentiment = analyze_sentiment(article)

    summary = summarize_article(article)

    return sentiment, summary


# ============================================================
# PDF GENERATOR
# ============================================================

def create_pdf(title, content):

    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    title_style = styles["Title"]
    body_style = styles["BodyText"]

    body_style.alignment = TA_LEFT
    body_style.leading = 15

    story = []

    story.append(
        Paragraph(title, title_style)
    )

    story.append(
        Spacer(1, 20)
    )

    paragraphs = content.split("\n")

    for paragraph in paragraphs:

        if paragraph.strip():

            safe_text = (
                paragraph
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )

            story.append(
                Paragraph(
                    safe_text,
                    body_style
                )
            )

            story.append(
                Spacer(1, 10)
            )

    document.build(story)

    buffer.seek(0)

    return buffer.getvalue()


# ============================================================
# HEADER
# ============================================================

st.title("Article Analyzer")

st.write(
    "Paste an article below and choose an analysis option."
)


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Sentiment Analysis",
        "Summarize",
        "Full Analysis",
        "Citation Generator",
        "Paraphraser"
    ]
)


# ============================================================
# SENTIMENT ANALYSIS TAB
# ============================================================

with tab1:

    st.header("Sentiment Analysis")

    st.write(
        "Analyze the sentiment of an article and view positive and negative confidence scores."
    )

    sentiment_input = st.text_area(
        "Article",
        placeholder="Paste your article here...",
        height=300,
        label_visibility="collapsed"
    )

    analyze_button = st.button(
        "Analyze Sentiment"
    )

    if analyze_button:

        if not sentiment_input.strip():

            st.warning("Please paste an article.")

        else:

            result = analyze_sentiment(
                sentiment_input
            )

            sentiment = result["sentiment"]
            confidence = result["confidence"]
            positive = result["positive"]
            negative = result["negative"]

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

            pdf_content = f"""
Overall Sentiment: {sentiment}

Confidence Score: {confidence:.2f}%

Positive Sentiment: {positive:.2f}%

Negative Sentiment: {negative:.2f}%
"""

            pdf = create_pdf(
                "Sentiment Analysis",
                pdf_content
            )

            st.download_button(
                "Download Sentiment Analysis PDF",
                data=pdf,
                file_name="sentiment_analysis.pdf",
                mime="application/pdf"
            )


# ============================================================
# SUMMARIZATION TAB
# ============================================================

with tab2:

    st.header("Article Summary")

    st.write(
        "Generate a clear and concise summary of your article."
    )

    summary_input = st.text_area(
        "Article",
        placeholder="Paste your article here...",
        height=300,
        label_visibility="collapsed"
    )

    summary_button = st.button(
        "Generate Summary"
    )

    if summary_button:

        if not summary_input.strip():

            st.warning(
                "Please paste an article."
            )

        else:

            with st.spinner(
                "Generating summary..."
            ):

                summary = summarize_article(
                    summary_input
                )

            st.markdown(summary)

            pdf = create_pdf(
                "Article Summary",
                summary
            )

            st.download_button(
                "Download Summary PDF",
                data=pdf,
                file_name="article_summary.pdf",
                mime="application/pdf"
            )


# ============================================================
# FULL ANALYSIS TAB
# ============================================================

with tab3:

    st.header("Full Article Analysis")

    st.write(
        "Perform both sentiment analysis and AI-powered article summarization."
    )

    full_input = st.text_area(
        "Article",
        placeholder="Paste your article here...",
        height=300,
        label_visibility="collapsed"
    )

    full_button = st.button(
        "Analyze Article"
    )

    if full_button:

        if not full_input.strip():

            st.warning(
                "Please paste an article."
            )

        else:

            with st.spinner(
                "Analyzing article..."
            ):

                sentiment, summary = full_analysis(
                    full_input
                )

            st.subheader(
                "Sentiment Analysis"
            )

            st.write(
                f"Overall Sentiment: {sentiment['sentiment']}"
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Confidence",
                    f"{sentiment['confidence']:.2f}%"
                )

            with col2:

                st.metric(
                    "Positive",
                    f"{sentiment['positive']:.2f}%"
                )

            with col3:

                st.metric(
                    "Negative",
                    f"{sentiment['negative']:.2f}%"
                )

            st.subheader(
                "Article Summary"
            )

            st.markdown(summary)

            pdf_content = f"""
SENTIMENT ANALYSIS

Overall Sentiment:
{sentiment['sentiment']}

Confidence Score:
{sentiment['confidence']:.2f}%

Positive:
{sentiment['positive']:.2f}%

Negative:
{sentiment['negative']:.2f}%


ARTICLE SUMMARY

{summary}
"""

            pdf = create_pdf(
                "Full Article Analysis",
                pdf_content
            )

            st.download_button(
                "Download Full Analysis PDF",
                data=pdf,
                file_name="full_article_analysis.pdf",
                mime="application/pdf"
            )


# ============================================================
# CITATION GENERATOR TAB
# ============================================================

with tab4:

    st.header("Citation Generator")

    st.write(
        "Generate an academic citation for an article."
    )

    citation_title = st.text_input(
        "Article Title"
    )

    citation_author = st.text_input(
        "Author"
    )

    citation_year = st.text_input(
        "Publication Year"
    )

    citation_url = st.text_input(
        "Article URL"
    )

    citation_style = st.selectbox(
        "Citation Style",
        [
            "APA 7",
            "MLA 9",
            "Harvard",
            "Chicago"
        ]
    )

    citation_button = st.button(
        "Generate Citation"
    )

    if citation_button:

        with st.spinner(
            "Generating citation..."
        ):

            citation = generate_citation(
                citation_title,
                citation_author,
                citation_year,
                citation_url,
                citation_style
            )

        st.subheader(
            "Generated Citation"
        )

        st.write(citation)

        pdf = create_pdf(
            "Generated Citation",
            citation
        )

        st.download_button(
            "Download Citation PDF",
            data=pdf,
            file_name="citation.pdf",
            mime="application/pdf"
        )


# ============================================================
# PARAPHRASER TAB
# ============================================================

with tab5:

    st.header("AI Paraphraser")

    st.write(
        "Rewrite text while preserving its original meaning."
    )

    paraphrase_input = st.text_area(
        "Text",
        placeholder="Paste your text here...",
        height=300,
        label_visibility="collapsed"
    )

    paraphrase_button = st.button(
        "Paraphrase Text"
    )

    if paraphrase_button:

        if not paraphrase_input.strip():

            st.warning(
                "Please paste text to paraphrase."
            )

        else:

            with st.spinner(
                "Paraphrasing..."
            ):

                paraphrased = paraphrase_article(
                    paraphrase_input
                )

            st.subheader(
                "Paraphrased Text"
            )

            st.markdown(
                paraphrased
            )

            pdf = create_pdf(
                "Paraphrased Text",
                paraphrased
            )

            st.download_button(
                "Download Paraphrased PDF",
                data=pdf,
                file_name="paraphrased_text.pdf",
                mime="application/pdf"
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Article Analyzer | Hugging Face + OpenRouter + Streamlit"
)
