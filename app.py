import os
from io import BytesIO

import requests
import streamlit as st
from transformers import pipeline

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer


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
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 48px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .main-description {
        font-size: 18px;
        margin-bottom: 30px;
    }

    .section-title {
        font-size: 32px;
        font-weight: 700;
        margin-top: 10px;
        margin-bottom: 8px;
    }

    .section-description {
        font-size: 16px;
        margin-bottom: 20px;
    }

    .sentiment-positive {
        padding: 18px;
        border-radius: 10px;
        background-color: #e8f7ee;
        color: #16803c;
        font-size: 20px;
        font-weight: 600;
        margin-top: 20px;
        margin-bottom: 20px;
    }

    .sentiment-negative {
        padding: 18px;
        border-radius: 10px;
        background-color: #fdecec;
        color: #c62828;
        font-size: 20px;
        font-weight: 600;
        margin-top: 20px;
        margin-bottom: 20px;
    }

    .result-box {
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #dddddd;
        background-color: #fafafa;
        margin-top: 20px;
        margin-bottom: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# API KEY
# ============================================================

def get_api_key():

    try:

        if "OPENROUTER_API_KEY" in st.secrets:
            return st.secrets["OPENROUTER_API_KEY"]

    except Exception:
        pass

    return os.environ.get("OPENROUTER_API_KEY")


# ============================================================
# LOAD HUGGING FACE SENTIMENT MODEL
# ============================================================

@st.cache_resource
def load_sentiment_model():

    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )


sentiment_pipeline = load_sentiment_model()


# ============================================================
# OPENROUTER AI
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
            "confidence": 0.0,
            "positive": 0.0,
            "negative": 0.0
        }

    try:

        # The current Hugging Face model has a 512-token limit.
        text = article[:512]

        results = sentiment_pipeline(
            text,
            top_k=2,
            truncation=True
        )

        if results and isinstance(results[0], list):
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
            "confidence": 0.0,
            "positive": 0.0,
            "negative": 0.0
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

Identify the main topic and explain the most important points.

Mention important benefits, opportunities, challenges, or risks
when they are present in the article.

Do not invent information.

Keep the summary easy to read.

Article:

{article}
"""

    return ask_ai(prompt)


# ============================================================
# PARAPHRASER
# ============================================================

def paraphrase_article(text):

    if not text or not text.strip():

        return "Please paste text to paraphrase."

    prompt = f"""
You are an AI paraphrasing assistant.

Rewrite the following text while preserving its original meaning.

Rules:

- Preserve the original meaning.
- Do not add information.
- Do not remove important information.
- Improve clarity and sentence structure.
- Use natural language.
- Keep the same language as the original text.

Text:

{text}
"""

    return ask_ai(prompt)


# ============================================================
# CITATION GENERATOR
# ============================================================

def generate_citation(
    title,
    author,
    year,
    url,
    style
):

    if not title.strip():

        return "Please enter the article title."

    prompt = f"""
Generate an academic citation using the information provided below.

Citation style:
{style}

Article title:
{title}

Author:
{author}

Publication year:
{year}

URL:
{url}

Instructions:

Generate the citation using only the information provided.

Do not invent missing information.

Follow the requested citation style.

Return only the citation.
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

    body_style.leading = 15

    story = []

    story.append(
        Paragraph(
            title,
            title_style
        )
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
# MAIN HEADER
# ============================================================

st.markdown(
    '<div class="main-title">Article Analyzer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-description">'
    'Paste an article below and choose an analysis option.'
    '</div>',
    unsafe_allow_html=True
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
# TAB 1 — SENTIMENT ANALYSIS
# ============================================================

with tab1:

    st.markdown(
        '<div class="section-title">Sentiment Analysis</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        'Analyze the sentiment of an article and view positive and negative confidence scores.'
        '</div>',
        unsafe_allow_html=True
    )

    sentiment_input = st.text_area(
        "Article",
        placeholder="Paste your article here...",
        height=300,
        label_visibility="collapsed",
        key="sentiment_article"
    )

    analyze_sentiment_button = st.button(
        "Analyze Sentiment",
        key="analyze_sentiment_button"
    )

    if analyze_sentiment_button:

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

            sentiment = result["sentiment"]
            confidence = result["confidence"]
            positive = result["positive"]
            negative = result["negative"]

            if sentiment == "POSITIVE":

                st.markdown(
                    f"""
                    <div class="sentiment-positive">
                    Overall Sentiment: {sentiment}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            elif sentiment == "NEGATIVE":

                st.markdown(
                    f"""
                    <div class="sentiment-negative">
                    Overall Sentiment: {sentiment}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.error(sentiment)

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

            pdf_data = create_pdf(
                "Sentiment Analysis",
                pdf_content
            )

            st.download_button(
                "Download Sentiment Analysis PDF",
                data=pdf_data,
                file_name="sentiment_analysis.pdf",
                mime="application/pdf",
                key="download_sentiment_pdf"
            )


# ============================================================
# TAB 2 — SUMMARIZE
# ============================================================

with tab2:

    st.markdown(
        '<div class="section-title">Article Summary</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        'Generate a clear and concise summary of your article.'
        '</div>',
        unsafe_allow_html=True
    )

    summary_input = st.text_area(
        "Article",
        placeholder="Paste your article here...",
        height=300,
        label_visibility="collapsed",
        key="summary_article"
    )

    summary_button = st.button(
        "Generate Summary",
        key="summary_button"
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

            st.markdown(
                summary
            )

            summary_pdf = create_pdf(
                "Article Summary",
                summary
            )

            st.download_button(
                "Download Summary PDF",
                data=summary_pdf,
                file_name="article_summary.pdf",
                mime="application/pdf",
                key="download_summary_pdf"
            )


# ============================================================
# TAB 3 — FULL ANALYSIS
# ============================================================

with tab3:

    st.markdown(
        '<div class="section-title">Full Article Analysis</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        'Perform sentiment analysis and generate an AI-powered article summary.'
        '</div>',
        unsafe_allow_html=True
    )

    full_input = st.text_area(
        "Article",
        placeholder="Paste your article here...",
        height=300,
        label_visibility="collapsed",
        key="full_analysis_article"
    )

    full_button = st.button(
        "Analyze Article",
        key="full_analysis_button"
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

            if sentiment:

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
                    "Sentiment Breakdown"
                )

                st.write(
                    f"Positive: {sentiment['positive']:.2f}%"
                )

                st.progress(
                    min(
                        sentiment["positive"] / 100,
                        1.0
                    )
                )

                st.write(
                    f"Negative: {sentiment['negative']:.2f}%"
                )

                st.progress(
                    min(
                        sentiment["negative"] / 100,
                        1.0
                    )
                )

            st.subheader(
                "Article Summary"
            )

            st.markdown(
                summary
            )

            full_pdf_content = f"""
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

            full_pdf = create_pdf(
                "Full Article Analysis",
                full_pdf_content
            )

            st.download_button(
                "Download Full Analysis PDF",
                data=full_pdf,
                file_name="full_article_analysis.pdf",
                mime="application/pdf",
                key="download_full_analysis_pdf"
            )


# ============================================================
# TAB 4 — CITATION GENERATOR
# ============================================================

with tab4:

    st.markdown(
        '<div class="section-title">Citation Generator</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        'Generate an academic citation for an article.'
        '</div>',
        unsafe_allow_html=True
    )

    citation_title = st.text_input(
        "Article Title",
        key="citation_title"
    )

    citation_author = st.text_input(
        "Author",
        key="citation_author"
    )

    citation_year = st.text_input(
        "Publication Year",
        key="citation_year"
    )

    citation_url = st.text_input(
        "Article URL",
        key="citation_url"
    )

    citation_style = st.selectbox(
        "Citation Style",
        [
            "APA 7",
            "MLA 9",
            "Harvard",
            "Chicago"
        ],
        key="citation_style"
    )

    citation_button = st.button(
        "Generate Citation",
        key="citation_button"
    )

    if citation_button:

        if not citation_title.strip():

            st.warning(
                "Please enter the article title."
            )

        else:

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

            st.markdown(
                citation
            )

            citation_pdf = create_pdf(
                "Generated Citation",
                citation
            )

            st.download_button(
                "Download Citation PDF",
                data=citation_pdf,
                file_name="citation.pdf",
                mime="application/pdf",
                key="download_citation_pdf"
            )


# ============================================================
# TAB 5 — PARAPHRASER
# ============================================================

with tab5:

    st.markdown(
        '<div class="section-title">AI Paraphraser</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        'Rewrite text while preserving its original meaning.'
        '</div>',
        unsafe_allow_html=True
    )

    paraphrase_input = st.text_area(
        "Text",
        placeholder="Paste your text here...",
        height=300,
        label_visibility="collapsed",
        key="paraphrase_text"
    )

    paraphrase_button = st.button(
        "Paraphrase Text",
        key="paraphrase_button"
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

            paraphrase_pdf = create_pdf(
                "Paraphrased Text",
                paraphrased
            )

            st.download_button(
                "Download Paraphrased PDF",
                data=paraphrase_pdf,
                file_name="paraphrased_text.pdf",
                mime="application/pdf",
                key="download_paraphrase_pdf"
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Article Analyzer | Hugging Face + OpenRouter + Streamlit"
)
