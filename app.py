
import os
import requests
import gradio as gr


# Get API key from environment variable
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "openai/gpt-4o-mini"


def ask_ai(prompt):
    if not OPENROUTER_API_KEY:
        return "Error: OPENROUTER_API_KEY is not set."

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
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

        response.raise_for_status()

        result = response.json()

        return result["choices"][0]["message"]["content"]

    except Exception as e:
        return f"Error: {str(e)}"


def summarize_article(article):
    if not article.strip():
        return "Please paste an article."

    prompt = f"""
Summarize the following article in clear and concise language.

Article:
{article}
"""

    return ask_ai(prompt)


def analyze_sentiment(article):
    if not article.strip():
        return "Please paste an article."

    prompt = f"""
Analyze the sentiment of the following article.

Return:
- Overall sentiment (Positive, Negative, or Neutral)
- A short explanation

Article:
{article}
"""

    return ask_ai(prompt)


def full_analysis(article):
    sentiment_result = analyze_sentiment(article)
    summary_result = summarize_article(article)

    return f"""
SENTIMENT ANALYSIS
==================

{sentiment_result}

ARTICLE SUMMARY
===============

{summary_result}
"""


# Build Gradio interface
with gr.Blocks() as demo:

    gr.Markdown(
        """
        # Article Analyzer

        Analyze articles using AI-powered sentiment analysis
        and summarization.
        """
    )

    # Summary Tab
    with gr.Tab("Summarize"):

        summary_input = gr.Textbox(
            label="Article",
            lines=15,
            placeholder="Paste your article here..."
        )

        summary_button = gr.Button("Summarize")

        summary_output = gr.Textbox(
            label="Summary",
            lines=10
        )

        summary_button.click(
            summarize_article,
            inputs=summary_input,
            outputs=summary_output
        )

    # Sentiment Analysis Tab
    with gr.Tab("Sentiment Analysis"):

        sentiment_input = gr.Textbox(
            label="Article",
            lines=15,
            placeholder="Paste your article here..."
        )

        sentiment_button = gr.Button("Analyze Sentiment")

        sentiment_output = gr.Textbox(
            label="Sentiment Result",
            lines=10
        )

        sentiment_button.click(
            analyze_sentiment,
            inputs=sentiment_input,
            outputs=sentiment_output
        )

    # Full Analysis Tab
    with gr.Tab("Full Analysis"):

        full_input = gr.Textbox(
            label="Article",
            lines=15,
            placeholder="Paste your article here..."
        )

        full_button = gr.Button("Analyze Article")

        full_output = gr.Textbox(
            label="Results",
            lines=15
        )

        full_button.click(
            full_analysis,
            inputs=full_input,
            outputs=full_output
        )


# Launch application
if __name__ == "__main__":

    if not os.environ.get("OPENROUTER_API_KEY"):
        print("Warning: OPENROUTER_API_KEY is not set!")

    port = int(os.environ.get("PORT", 7860))

    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        debug=True
    )
