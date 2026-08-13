import os
import requests
import streamlit as st
from transformers import pipeline
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import simpleSplit


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

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

API_URL = "https://openrouter.ai/api/v1/chat/completions"

MODEL = "openrouter/free"


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .main-description {
        font-size: 18px;
        margin-bottom: 30px;
    }

    .section-description {
        font-size: 16px;
        margin-bottom: 20px;
    }

    .sentiment-positive {
        padding: 15px;
        border-radius: 8px;
        background-color: #e8f7ee;
        border: 1px solid #b7e4c7;
        font-size: 18px;
    }

    .sentiment-negative {
        padding: 15px;
        border-radius: 8px;
        background-color: #fdecec;
        border: 1px solid #f5b5b5;
        font-size: 18px;
    }

    .score-box {
        padding: 15px;
        border-radius: 8px;
        background-color: #f5f6f8;
        text-align: center;
    }

    .score-number {
        font-size: 30px;
        font-weight: 600;
    }

    .stButton > button {
        border-radius: 6px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# OPENROUTER AI FUNCTION
# ============================================================

def ask_ai(prompt):

    api_key = os.environ.get("OPENROUTER_API_KEY")

    if not api_key:
        return "Error: OPENROUTER_API_KEY is not set."

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://streamlit.io",
        "X-Title": "Article Analyzer"
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

        if "choices" not in result:
            return "Unable to generate a response."

        return result["choices"][0]["message"]["content"]

    except requests.exceptions.RequestException as e:

        return f"Request Error: {str(e)}"

    except Exception as e:

        return f"Error: {str(e)}"


# ============================================================
# MULTILINGUAL SENTIMENT MODEL
# ============================================================

@st.cache_resource
def load_sentiment_model():

    return pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-xlm-roberta-base-sentiment"
    )


try:

    sentiment_pipeline = load_sentiment_model()

except Exception as e:

    sentiment_pipeline = None


# ============================================================
# SENTIMENT ANALYSIS
# ============================================================

def analyze_sentiment(article):

    if not article or not article.strip():

        return {
            "sentiment": "No input",
            "confidence": 0,
            "positive": 0,
            "negative": 0,
            "error": "Please paste an article."
        }

    if sentiment_pipeline is None:

        return {
            "sentiment": "Error",
            "confidence": 0,
            "positive": 0,
            "negative": 0,
            "error": "Sentiment model could not be loaded."
        }

    try:

        text = article[:512]

        results = sentiment_pipeline(
            text,
            top_k=None,
            truncation=True
        )

        if results and isinstance(results[0], list):
            results = results[0]

        positive = 0.0
        negative = 0.0
        neutral = 0.0

        for result in results:

            label = result["label"].lower()
            score = result["score"]

            if "positive" in label:

                positive = score

            elif "negative" in label:

                negative = score

            elif "neutral" in label:

                neutral = score

        # ----------------------------------------------------
        # Convert to percentages
        # We display only positive and negative.
        # Neutral is redistributed proportionally.
        # ----------------------------------------------------

        total_polar = positive + negative

        if total_polar > 0:

            positive_percentage = (
                positive / total_polar
            ) * 100

            negative_percentage = (
                negative / total_polar
            ) * 100

        else:

            positive_percentage = 50
            negative_percentage = 50

        if positive_percentage >= negative_percentage:

            overall_sentiment = "POSITIVE"
            confidence = positive_percentage

        else:

            overall_sentiment = "NEGATIVE"
            confidence = negative_percentage

        return {
            "sentiment": overall_sentiment,
            "confidence": confidence,
            "positive": positive_percentage,
            "negative": negative_percentage,
            "error": None
        }

    except Exception as e:

        return {
            "sentiment": "Error",
            "confidence": 0,
            "positive": 0,
            "negative": 0,
            "error": str(e)
        }


# ============================================================
# ARTICLE SUMMARIZATION
# ============================================================

def summarize_article(article):

    if not article or not article.strip():

        return "Please paste an article."

    prompt = f"""
Summarize the following article clearly and concisely.

Write the summary in the same language as the original article.

Requirements:

- Identify the main topic.
- Highlight the most important points.
- Mention important benefits or opportunities.
- Mention important challenges or risks.
- Do not invent information.
- Keep the summary easy to read.
- Do not mention these instructions.
- Return only the summary.

Article:

{article}
"""

    return ask_ai(prompt)


# ============================================================
# AI PARAPHRASER
# ============================================================

def paraphrase_text(text, style):

    if not text or not text.strip():

        return "Please enter text to paraphrase."

    prompt = f"""
Paraphrase the following text while preserving its original meaning.

Write the paraphrased version in the same language as the original text.

Style: {style}

Requirements:

- Preserve the original meaning.
- Do not add information.
- Do not remove important information.
- Improve clarity and readability.
- Avoid copying the original wording unnecessarily.
- Return only the paraphrased text.

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
    publication,
    date,
    url,
    style
):

    if not title.strip():

        return "Please enter the article title."

    prompt = f"""
Generate a citation for the following source.

Citation style: {style}

Article title:
{title}

Author:
{author}

Publication/Website:
{publication}

Publication date:
{date}

URL:
{url}

Requirements:

- Use the requested citation style.
- Do not invent missing information.
- If information is missing, simply omit it.
- Return the citation only.
"""

    return ask_ai(prompt)


# ============================================================
# PDF GENERATOR
# ============================================================

def create_pdf(title, content):

    file_path = "article_analysis.pdf"

    pdf = canvas.Canvas(
        file_path,
        pagesize=A4
    )

    width, height = A4

    margin = 50

    y = height - margin

    pdf.setFont(
        "Helvetica-Bold",
        18
    )

    pdf.drawString(
        margin,
        y,
        title
    )

    y -= 35

    pdf.setFont(
        "Helvetica",
        10
    )

    # Convert content into lines
    paragraphs = content.split("\n")

    for paragraph in paragraphs:

        if not paragraph.strip():

            y -= 10
            continue

        lines = simpleSplit(
            paragraph,
            "Helvetica",
            10,
            width - (2 * margin)
        )

        for line in lines:

            if y < margin:

                pdf.showPage()

                pdf.setFont(
                    "Helvetica",
                    10
                )

                y = height - margin

            pdf.drawString(
                margin,
                y,
                line
            )

            y -= 15

    pdf.save()

    return file_path


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">Article Analyzer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-description">Paste an article below and choose an analysis option.</div>',
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
        "AI Paraphraser"
    ]
)


# ============================================================
# TAB 1 - SENTIMENT ANALYSIS
# ============================================================

with tab1:

    st.header("Sentiment Analysis")

    st.write(
        "Analyze the sentiment of an article and view positive and negative confidence scores."
    )

    article_sentiment = st.text_area(
        "Article",
        height=300,
        placeholder="Paste your article here..."
    )

    if st.button(
        "Analyze Sentiment",
        key="sentiment_button"
    ):

        if not article_sentiment.strip():

            st.warning("Please paste an article.")

        else:

            with st.spinner("Analyzing sentiment..."):

                result = analyze_sentiment(
                    article_sentiment
                )

            if result["error"]:

                st.error(
                    result["error"]
                )

            else:

                sentiment = result["sentiment"]

                if sentiment == "POSITIVE":

                    st.markdown(
                        f"""
                        <div class="sentiment-positive">
                        Overall Sentiment: <strong>POSITIVE</strong>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                else:

                    st.markdown(
                        f"""
                        <div class="sentiment-negative">
                        Overall Sentiment: <strong>NEGATIVE</strong>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                st.write("")

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
                    f"Positive: {result['positive']:.2f}%"
                )

                st.progress(
                    int(result["positive"])
                )

                st.write(
                    f"Negative: {result['negative']:.2f}%"
                )

                st.progress(
                    int(result["negative"])
                )

                pdf_content = f"""
Overall Sentiment: {result['sentiment']}

Confidence Score: {result['confidence']:.2f}%

Positive: {result['positive']:.2f}%

Negative: {result['negative']:.2f}%
"""

                pdf_file = create_pdf(
                    "Sentiment Analysis",
                    pdf_content
                )

                with open(
                    pdf_file,
                    "rb"
                ) as file:

                    st.download_button(
                        "Download Sentiment Analysis PDF",
                        file,
                        file_name="sentiment_analysis.pdf",
                        mime="application/pdf"
                    )


# ============================================================
# TAB 2 - SUMMARIZE
# ============================================================

with tab2:

    st.header("Article Summary")

    st.write(
        "Generate a clear and concise summary of any article."
    )

    article_summary = st.text_area(
        "Article",
        height=300,
        placeholder="Paste your article here...",
        key="summary_article"
    )

    if st.button(
        "Generate Summary",
        key="summary_button"
    ):

        if not article_summary.strip():

            st.warning(
                "Please paste an article."
            )

        else:

            with st.spinner(
                "Generating summary..."
            ):

                summary = summarize_article(
                    article_summary
                )

            st.markdown(
                summary
            )

            pdf_file = create_pdf(
                "Article Summary",
                summary
            )

            with open(
                pdf_file,
                "rb"
            ) as file:

                st.download_button(
                    "Download Summary PDF",
                    file,
                    file_name="article_summary.pdf",
                    mime="application/pdf"
                )


# ============================================================
# TAB 3 - FULL ANALYSIS
# ============================================================

with tab3:

    st.header("Full Article Analysis")

    st.write(
        "Get sentiment analysis and an AI-powered summary in one result."
    )

    full_article = st.text_area(
        "Article",
        height=300,
        placeholder="Paste your article here...",
        key="full_article"
    )

    if st.button(
        "Analyze Article",
        key="full_button"
    ):

        if not full_article.strip():

            st.warning(
                "Please paste an article."
            )

        else:

            with st.spinner(
                "Analyzing article..."
            ):

                sentiment = analyze_sentiment(
                    full_article
                )

                summary = summarize_article(
                    full_article
                )

            st.subheader(
                "Sentiment Analysis"
            )

            if sentiment["sentiment"] == "POSITIVE":

                st.success(
                    f"Overall Sentiment: POSITIVE"
                )

            else:

                st.error(
                    f"Overall Sentiment: NEGATIVE"
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

            st.markdown(
                summary
            )

            full_pdf_content = f"""
ARTICLE ANALYSIS

SENTIMENT ANALYSIS

Overall Sentiment:
{sentiment['sentiment']}

Confidence:
{sentiment['confidence']:.2f}%

Positive:
{sentiment['positive']:.2f}%

Negative:
{sentiment['negative']:.2f}%


ARTICLE SUMMARY

{summary}
"""

            pdf_file = create_pdf(
                "Full Article Analysis",
                full_pdf_content
            )

            with open(
                pdf_file,
                "rb"
            ) as file:

                st.download_button(
                    "Download Full Analysis PDF",
                    file,
                    file_name="full_article_analysis.pdf",
                    mime="application/pdf"
                )


# ============================================================
# TAB 4 - CITATION GENERATOR
# ============================================================

with tab4:

    st.header("Citation Generator")

    st.write(
        "Generate properly formatted citations for your sources."
    )

    citation_style = st.selectbox(
        "Citation Style",
        [
            "APA 7th Edition",
            "MLA 9th Edition",
            "Chicago"
        ]
    )

    citation_title = st.text_input(
        "Article Title",
        placeholder="Enter the article title..."
    )

    citation_author = st.text_input(
        "Author",
        placeholder="Enter the author's name..."
    )

    citation_publication = st.text_input(
        "Publication / Website",
        placeholder="Enter the publication or website..."
    )

    citation_date = st.text_input(
        "Publication Date",
        placeholder="Enter the publication date..."
    )

    citation_url = st.text_input(
        "URL",
        placeholder="Paste the article URL..."
    )

    if st.button(
        "Generate Citation",
        key="citation_button"
    ):

        with st.spinner(
            "Generating citation..."
        ):

            citation = generate_citation(
                citation_title,
                citation_author,
                citation_publication,
                citation_date,
                citation_url,
                citation_style
            )

        st.subheader(
            "Generated Citation"
        )

        st.write(
            citation
        )

        pdf_file = create_pdf(
            "Citation",
            citation
        )

        with open(
            pdf_file,
            "rb"
        ) as file:

            st.download_button(
                "Download Citation PDF",
                file,
                file_name="citation.pdf",
                mime="application/pdf"
            )


# ============================================================
# TAB 5 - AI PARAPHRASER
# ============================================================

with tab5:

    st.header("AI Paraphraser")

    st.write(
        "Rewrite text while preserving its original meaning."
    )

    paraphrase_text_input = st.text_area(
        "Text",
        height=300,
        placeholder="Paste the text you want to paraphrase here..."
    )

    paraphrase_style = st.selectbox(
        "Writing Style",
        [
            "Professional",
            "Academic",
            "Simple",
            "Formal",
            "Natural"
        ]
    )

    if st.button(
        "Paraphrase Text",
        key="paraphrase_button"
    ):

        if not paraphrase_text_input.strip():

            st.warning(
                "Please enter text to paraphrase."
            )

        else:

            with st.spinner(
                "Paraphrasing text..."
            ):

                paraphrased = paraphrase_text(
                    paraphrase_text_input,
                    paraphrase_style
                )

            st.subheader(
                "Paraphrased Text"
            )

            st.markdown(
                paraphrased
            )

            pdf_file = create_pdf(
                "Paraphrased Text",
                paraphrased
            )

            with open(
                pdf_file,
                "rb"
            ) as file:

                st.download_button(
                    "Download Paraphrased Text PDF",
                    file,
                    file_name="paraphrased_text.pdf",
                    mime="application/pdf"
                )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Article Analyzer | Sentiment Analysis | Summarization | Citation Generator | AI Paraphraser"
)
