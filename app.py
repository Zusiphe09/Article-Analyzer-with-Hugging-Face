import os
import html
from io import BytesIO

import requests
import streamlit as st
from transformers import pipeline

from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer
)
from reportlab.lib.styles import (
    getSampleStyleSheet,
    ParagraphStyle
)
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import mm


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
# GET OPENROUTER API KEY
# ============================================================

def get_api_key():

    # Streamlit Cloud Secrets
    try:

        api_key = st.secrets["OPENROUTER_API_KEY"]

        if api_key:
            return api_key

    except Exception:
        pass

    # Local environment variable
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

Return ONLY the language name.

Do not provide an explanation.

Article:

{article[:3000]}
"""

    result = ask_ai(prompt)

    if result.startswith("Error"):

        return "Unknown"

    return result.strip()


# ============================================================
# SENTIMENT ANALYSIS
# ============================================================

def analyze_sentiment(article):

    if not article or not article.strip():

        return {
            "error": "Please paste an article."
        }

    try:

        language = detect_language(article)

        # ----------------------------------------------------
        # English
        # ----------------------------------------------------

        if language.lower() == "english":

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
        # Other languages
        # ----------------------------------------------------

        else:

            prompt = f"""
Analyze the sentiment of this article.

The article may be written in any language.

Only use these two sentiment categories:

POSITIVE
NEGATIVE

Do NOT return NEUTRAL.

Return exactly:

Overall Sentiment: POSITIVE
Positive Score: 75.00
Negative Score: 25.00
Confidence Score: 75.00

The positive and negative scores should add up to
approximately 100.

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
# AI PARAPHRASER
# ============================================================

def paraphrase_article(article, style):

    if not article or not article.strip():

        return "Please paste text to paraphrase."

    style_instructions = {

        "Standard": """
Rewrite the text naturally while preserving the original
meaning, facts, and important details.
""",

        "Simple": """
Rewrite the text using simple, clear language that is easy
for a general reader to understand.
""",

        "Professional": """
Rewrite the text using professional and polished language
suitable for business or workplace communication.
""",

        "Academic": """
Rewrite the text using formal academic language while
preserving the original meaning and factual information.
Do not add unsupported information.
""",

        "Concise": """
Rewrite the text more concisely while preserving the
important meaning and key information.
"""
    }

    selected_instruction = style_instructions.get(
        style,
        style_instructions["Standard"]
    )

    prompt = f"""
You are an AI paraphrasing assistant.

Paraphrase the following text.

{selected_instruction}

The text may be written in any language.

Keep the paraphrased version in the SAME LANGUAGE as the
original text.

Do not translate the text.

Do not add new facts.

Do not remove important information.

Do not mention that the text was paraphrased.

Return only the paraphrased text.

Original text:

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

    if style == "APA 7":

        citation = (
            f"{author}. ({date}). "
            f"{title}. {publication}. {url}"
        )

    elif style == "MLA 9":

        citation = (
            f'{author}. "{title}." '
            f"{publication}, {date}, {url}."
        )

    elif style == "Harvard":

        citation = (
            f"{author} ({date}) '{title}', "
            f"{publication}. Available at: {url}."
        )

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
# PDF GENERATOR
# ============================================================

def create_pdf(title, content):

    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=20 * mm,
        leftMargin=20 * mm,
        topMargin=20 * mm,
        bottomMargin=20 * mm
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "PDFTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        spaceAfter=20
    )

    heading_style = ParagraphStyle(
        "PDFHeading",
        parent=styles["Heading2"],
        spaceBefore=12,
        spaceAfter=8
    )

    body_style = ParagraphStyle(
        "PDFBody",
        parent=styles["BodyText"],
        leading=15,
        spaceAfter=8
    )

    story = []

    story.append(
        Paragraph(
            html.escape(title),
            title_style
        )
    )

    for line in content.splitlines():

        line = line.strip()

        if not line:

            story.append(
                Spacer(1, 6)
            )

            continue

        clean_line = line.replace(
            "**",
            ""
        )

        if (
            clean_line.isupper()
            and len(clean_line) < 80
        ):

            story.append(
                Paragraph(
                    html.escape(clean_line),
                    heading_style
                )
            )

        else:

            story.append(
                Paragraph(
                    html.escape(clean_line),
                    body_style
                )
            )

    document.build(story)

    buffer.seek(0)

    return buffer.getvalue()


# ============================================================
# SENTIMENT PDF
# ============================================================

def create_sentiment_pdf(result):

    content = f"""
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

    return create_pdf(
        "Article Sentiment Analysis",
        content
    )


# ============================================================
# SUMMARY PDF
# ============================================================

def create_summary_pdf(summary):

    return create_pdf(
        "Article Summary",
        summary
    )


# ============================================================
# PARAPHRASED TEXT PDF
# ============================================================

def create_paraphrase_pdf(
    paraphrased_text,
    style
):

    content = f"""
Paraphrasing Style:
{style}

PARAPHRASED TEXT

{paraphrased_text}
"""

    return create_pdf(
        "Paraphrased Article",
        content
    )


# ============================================================
# FULL ANALYSIS PDF
# ============================================================

def create_full_analysis_pdf(
    sentiment_result,
    summary_result
):

    content = f"""
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
{sentiment_result["negative"]:.2f}


ARTICLE SUMMARY

{summary_result}
"""

    return create_pdf(
        "Full Article Analysis",
        content
    )


# ============================================================
# CITATION PDF
# ============================================================

def create_citation_pdf(
    citation,
    style
):

    content = f"""
Citation Style:
{style}

GENERATED CITATION

{citation}
"""

    return create_pdf(
        "Article Citation",
        content
    )


# ============================================================
# APPLICATION HEADER
# ============================================================

st.title("Article Analyzer")

st.markdown(
    "Paste an article below and choose an analysis option."
)


# ============================================================
# APPLICATION TABS
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

                st.caption(
                    f"Detected Language: "
                    f"{result['language']}"
                )

                if result["sentiment"] == "POSITIVE":

                    st.success(
                        "Overall Sentiment: POSITIVE"
                    )

                else:

                    st.error(
                        "Overall Sentiment: NEGATIVE"
                    )

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.metric(
                        "Confidence",
                        f"{result['confidence']:.2f}%"
                    )

                with col2:

                    st.metric(
                        "Positive",
                        f"{result['positive']:.2f}%"
                    )

                with col3:

                    st.metric(
                        "Negative",
                        f"{result['negative']:.2f}%"
                    )

                st.subheader(
                    "Sentiment Breakdown"
                )

                st.write(
                    f"Positive: "
                    f"{result['positive']:.2f}%"
                )

                st.progress(
                    int(result["positive"])
                )

                st.write(
                    f"Negative: "
                    f"{result['negative']:.2f}%"
                )

                st.progress(
                    int(result["negative"])
                )

                sentiment_pdf = create_sentiment_pdf(
                    result
                )

                st.download_button(
                    label="Download Sentiment Analysis PDF",
                    data=sentiment_pdf,
                    file_name="sentiment_analysis.pdf",
                    mime="application/pdf",
                    key="download_sentiment_pdf"
                )


# ============================================================
# SUMMARY
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

            st.markdown(
                summary_result
            )

            summary_pdf = create_summary_pdf(
                summary_result
            )

            st.download_button(
                label="Download Summary PDF",
                data=summary_pdf,
                file_name="article_summary.pdf",
                mime="application/pdf",
                key="download_summary_pdf"
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

            with st.spinner(
                "Analyzing article..."
            ):

                sentiment_result = analyze_sentiment(
                    full_input
                )

            if "error" in sentiment_result:

                st.error(
                    sentiment_result["error"]
                )

            else:

                with st.spinner(
                    "Generating summary..."
                ):

                    summary_result = summarize_article(
                        full_input
                    )

                st.caption(
                    f"Detected Language: "
                    f"{sentiment_result['language']}"
                )

                st.divider()

                st.header(
                    "Sentiment Analysis"
                )

                if (
                    sentiment_result["sentiment"]
                    == "POSITIVE"
                ):

                    st.success(
                        "Overall Sentiment: POSITIVE"
                    )

                else:

                    st.error(
                        "Overall Sentiment: NEGATIVE"
                    )

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.metric(
                        "Confidence",
                        f"{sentiment_result['confidence']:.2f}%"
                    )

                with col2:

                    st.metric(
                        "Positive",
                        f"{sentiment_result['positive']:.2f}%"
                    )

                with col3:

                    st.metric(
                        "Negative",
                        f"{sentiment_result['negative']:.2f}%"
                    )

                st.subheader(
                    "Sentiment Breakdown"
                )

                st.write(
                    f"Positive: "
                    f"{sentiment_result['positive']:.2f}%"
                )

                st.progress(
                    int(sentiment_result["positive"])
                )

                st.write(
                    f"Negative: "
                    f"{sentiment_result['negative']:.2f}%"
                )

                st.progress(
                    int(sentiment_result["negative"])
                )

                st.divider()

                st.header(
                    "Article Summary"
                )

                st.markdown(
                    summary_result
                )

                full_pdf = create_full_analysis_pdf(
                    sentiment_result,
                    summary_result
                )

                st.download_button(
                    label="Download Full Analysis PDF",
                    data=full_pdf,
                    file_name="full_article_analysis.pdf",
                    mime="application/pdf",
                    key="download_full_pdf"
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

            citation_pdf = create_citation_pdf(
                citation,
                style
            )

            st.download_button(
                label="Download Citation PDF",
                data=citation_pdf,
                file_name="article_citation.pdf",
                mime="application/pdf",
                key="download_citation_pdf"
            )


# ============================================================
# PARAPHRASER
# ============================================================

with tab5:

    st.subheader("AI Paraphraser")

    st.markdown(
        "Rewrite text while preserving its original meaning."
    )

    paraphrase_input = st.text_area(
        "Text",
        height=300,
        placeholder="Paste the text you want to paraphrase...",
        key="paraphrase_input"
    )

    paraphrase_style = st.selectbox(
        "Paraphrasing Style",
        [
            "Standard",
            "Simple",
            "Professional",
            "Academic",
            "Concise"
        ]
    )

    if st.button(
        "Paraphrase Text",
        key="paraphrase_button"
    ):

        if not paraphrase_input.strip():

            st.warning(
                "Please paste text to paraphrase."
            )

        else:

            with st.spinner(
                "Paraphrasing text..."
            ):

                paraphrased_result = paraphrase_article(
                    paraphrase_input,
                    paraphrase_style
                )

            st.subheader(
                "Paraphrased Text"
            )

            st.markdown(
                paraphrased_result
            )

            paraphrase_pdf = create_paraphrase_pdf(
                paraphrased_result,
                paraphrase_style
            )

            st.download_button(
                label="Download Paraphrased Text PDF",
                data=paraphrase_pdf,
                file_name="paraphrased_article.pdf",
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
