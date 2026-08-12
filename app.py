
import os
import requests
import gradio as gr
from transformers import pipeline


# ============================================================
# CONFIGURATION
# ============================================================

# Get OpenRouter API key from environment variable
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

# OpenRouter API
API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Free OpenRouter model for summarization
MODEL = "openrouter/free"


# ============================================================
# HUGGING FACE SENTIMENT ANALYSIS MODEL
# ============================================================

sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
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

        # Show the actual OpenRouter error
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
        return (
            "Please paste an article.",
            "",
            ""
        )

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

        # Determine overall sentiment
        if positive_score >= negative_score:
            sentiment = "POSITIVE"
            confidence = positive_score
        else:
            sentiment = "NEGATIVE"
            confidence = negative_score

        # Create visual bars
        positive_blocks = int(positive_score / 5)
        negative_blocks = int(negative_score / 5)

        positive_bar = (
            "█" * positive_blocks +
            "░" * (20 - positive_blocks)
        )

        negative_bar = (
            "█" * negative_blocks +
            "░" * (20 - negative_blocks)
        )

        # Sentiment result
        sentiment_result = f"""
## Overall Sentiment: **{sentiment}**

### Confidence Score: **{confidence:.2f}%**
"""

        # Percentage results
        sentiment_percentages = f"""
### Sentiment Scores

**Positive: {positive_score:.2f}%**

**Negative: {negative_score:.2f}%**
"""

        # Visual representation
        sentiment_visual = f"""
### Sentiment Breakdown

**Positive**

`{positive_bar}`

**Negative**

`{negative_bar}`
"""

        return (
            sentiment_result,
            sentiment_percentages,
            sentiment_visual
        )

    except Exception as e:

        return (
            f"Error: {str(e)}",
            "",
            ""
        )
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

    if not article or not article.strip():
        return "Please paste an article."

    # Run sentiment analysis
    sentiment_result, percentages, visual = analyze_sentiment(article)

    # Generate article summary
    summary_result = summarize_article(article)

    # Combine sentiment and summary
    return f"""
# ARTICLE ANALYSIS

---

## SENTIMENT ANALYSIS

{sentiment_result}

{percentages}

{visual}

---

## ARTICLE SUMMARY

{summary_result}
"""


# ============================================================
# GRADIO INTERFACE
# ============================================================

with gr.Blocks(
    title="Article Analyzer"
) as demo:

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    gr.Markdown(
        """
# 📰 Article Analyzer

Analyze articles using **AI-powered sentiment analysis
and summarization**.

Paste an article below and choose an analysis option.
"""
    )


    # ========================================================
    # SENTIMENT ANALYSIS TAB
    # ========================================================

    with gr.Tab("Sentiment Analysis"):

        gr.Markdown(
            """
### Sentiment Analysis

The application uses a pre-trained Hugging Face model
to identify the sentiment of the article and provide
a confidence score.
"""
        )

        sentiment_input = gr.Textbox(
            label="Article",
            lines=15,
            placeholder="Paste your article here..."
        )

        sentiment_button = gr.Button(
            "Analyze Sentiment"
        )

        sentiment_result = gr.Markdown(
            label="Sentiment Result"
        )

        sentiment_percentages = gr.Markdown(
            label="Sentiment Percentages"
        )

        sentiment_visual = gr.Markdown(
            label="Confidence Breakdown"
        )

        sentiment_button.click(
            analyze_sentiment,
            inputs=sentiment_input,
            outputs=[
                sentiment_result,
                sentiment_percentages,
                sentiment_visual
            ]
        )


    # ========================================================
    # SUMMARY TAB
    # ========================================================

    with gr.Tab("Summarize"):

        gr.Markdown(
            """
### Article Summarization

OpenRouter is used to generate a clear and detailed
summary of the article.
"""
        )

        summary_input = gr.Textbox(
            label="Article",
            lines=15,
            placeholder="Paste your article here..."
        )

        summary_button = gr.Button(
            "Generate Summary"
        )

        summary_output = gr.Markdown(
            label="Summary"
        )

        summary_button.click(
            summarize_article,
            inputs=summary_input,
            outputs=summary_output
        )


    # ========================================================
    # FULL ANALYSIS TAB
    # ========================================================

    with gr.Tab("Full Analysis"):

        gr.Markdown(
            """
### Full Article Analysis

This option combines sentiment analysis and
AI-powered summarization into one result.
"""
        )

        full_input = gr.Textbox(
            label="Article",
            lines=15,
            placeholder="Paste your article here..."
        )

        full_button = gr.Button(
            "Analyze Article"
        )

        full_output = gr.Markdown(
            label="Analysis Results"
        )

        full_button.click(
            full_analysis,
            inputs=full_input,
            outputs=full_output
        )


    # ========================================================
    # FOOTER
    # ========================================================

    gr.Markdown(
        """
---

**Article Analyzer | Hugging Face + OpenRouter + Gradio**
"""
    )


# ============================================================
# LAUNCH APPLICATION
# ============================================================

if __name__ == "__main__":

    if not os.environ.get("OPENROUTER_API_KEY"):
        print("Warning: OPENROUTER_API_KEY is not set!")

    port = int(
        os.environ.get(
            "PORT",
            7860
        )
    )

    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        debug=True
    )
