
import gradio as gr
import os
from transformers import pipeline
import requests
import json

# Initialize sentiment analysis pipeline with explicit model
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)

def summarize_text(text):
    """Summarize text using OpenRouter API"""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    
    if not api_key:
        return "Error: OPENROUTER_API_KEY not set. Please set it in your environment."
    
    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "openrouter/free",
                "messages": [
                    {
                        "role": "user",
                        "content": f"Please summarize the following text concisely:\n\n{text}"
                    }
                ],
                "temperature": 0.7,
                "max_tokens": 200
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            return f"API Error: {response.status_code} - {response.text}"
            
    except Exception as e:
        return f"Error: {str(e)}"

def analyze_sentiment(text):
    """Analyze sentiment using Hugging Face pipeline"""
    if not text.strip():
        return "Please enter some text to analyze."
    
    result = sentiment_pipeline(text)[0]
    return f"{result['label']} (Confidence: {result['score']:.2%})"


def full_analysis(text):
    """Run both sentiment analysis and summarization."""
    if not text.strip():
        return "Please enter some text to analyze."
    
    # 1. Call your Hugging Face sentiment function
    sentiment_result = analyze_sentiment(text)
    
    # 2. Call your custom OpenRouter summarization function (with correct name!)
    summary_result = summarize_text(text)
    
    # 3. Return both results with beautiful formatting
    return f"--- SENTIMENT ---\n{sentiment_result}\n\n--- SUMMARY ---\n{summary_result}"


# Create the Gradio interface
with gr.Blocks(title="Article Analyzer") as demo:
    gr.Markdown("# 📝 Article Analyzer")
    gr.Markdown("Analyze articles with summarization and sentiment analysis")
    
    with gr.Tab("Summarize"):
        gr.Markdown("### 📄 Text Summarization")
        with gr.Row():
            input_text = gr.Textbox(
                label="Enter text to summarize",
                placeholder="Paste your article or text here...",
                lines=10
            )
        with gr.Row():
            summarize_btn = gr.Button("📝 Summarize", variant="primary")
        output_summary = gr.Textbox(
            label="Summary",
            lines=5,
            interactive=False
        )
        summarize_btn.click(
            fn=summarize_text,
            inputs=input_text,
            outputs=output_summary
        )
    
    with gr.Tab("Sentiment Analysis"):
        gr.Markdown("### 🎯 Sentiment Analysis")
        with gr.Row():
            sentiment_input = gr.Textbox(
                label="Enter text for sentiment analysis",
                placeholder="Paste text to analyze sentiment...",
                lines=10
            )
        with gr.Row():
            sentiment_btn = gr.Button("🔍 Analyze Sentiment", variant="primary")
        output_sentiment = gr.Textbox(
            label="Sentiment Result",
            lines=3,
            interactive=False
        )
        sentiment_btn.click(
            fn=analyze_sentiment,
            inputs=sentiment_input,
            outputs=output_sentiment
        ) 

        
    with gr.Tab("Full Analysis"):
        gr.Markdown("### 📊 Combined Analysis")
        with gr.Row():
            full_input = gr.Textbox(
                label="Article Text",
                placeholder="Paste your article here for full analysis...",
                lines=10
            )
        with gr.Row():
            full_btn = gr.Button("Run Full Analysis", variant="primary")
        output_full = gr.Textbox(
            label="Combined Results",
            lines=12,              # Give it extra space to show both results comfortably!
            interactive=False
        )
        full_btn.click(
            fn=full_analysis,
            inputs=full_input,
            outputs=output_full
        )


if __name__ == "__main__":
2
if not os.environ.get("OPENROUTER_API_KEY"):
3
print("Warning: OPENROUTER_API_KEY is not set!")
4
 
5
port = int(os.environ.get("PORT", 7860))
6
 
7
demo.launch(
8
server_name="0.0.0.0",
9
server_port=port,
10
debug=True
11
)
