<img src="https://cdn.prod.website-files.com/677c400686e724409a5a7409/6790ad949cf622dc8dcd9fe4_nextwork-logo-leather.svg" alt="NextWork" width="300" />

# Article Analyzer with Hugging Face

**Project Link:** [View Project](https://nextwork.ai/projects/6f6b9ea8-b2ca-434c-98e0-79cf1a7c5135)

**Author:** Inga Nguse  
**Email:** inganguse09@gmail.com

---

![Image](https://nextwork.ai/proud_blue_vibrant_quince/uploads/6f6b9ea8-b2ca-434c-98e0-79cf1a7c5135_w329d37d)

## Project Overview

### Goals and approach

In this step, I'm setting up a Python development environment with the required packages and an OpenRouter API key so that I can build and run an AI-powered Article Analyzer that performs sentiment analysis and article summarization.

![Image](https://nextwork.ai/proud_blue_vibrant_quince/uploads/6f6b9ea8-b2ca-434c-98e0-79cf1a7c5135_6yt5fwf1)

### Securing the OpenRouter API key

I stored my key by creating a .env file and placing my OpenRouter API key in the OPENROUTER_API_KEY environment variable. This keeps the key separate from my source code and helps protect it from being exposed publicly.

## Building Sentiment Analysis with Hugging Face

### Local ML inference with Transformers

In this step, I'm building a sentiment analysis feature using a pre-trained Hugging Face model so that I can automatically analyze text and identify its emotional tone with a confidence score.

![Image](https://nextwork.ai/proud_blue_vibrant_quince/uploads/6f6b9ea8-b2ca-434c-98e0-79cf1a7c5135_aulhsqnh)

### How the sentiment pipeline processes text

The pipeline takes my text and processes it through a pre-trained sentiment analysis model before returning a sentiment label and confidence score.

## Discovering the Limits of Local Summarization

### Testing local summarization

In this step, I'm adding a local summarization pipeline to my app so that I can generate summaries of long articles and compare the performance of local AI on summarization versus sentiment analysis.

### Comparing local model quality to sentiment analysis

I noticed the sentiment result was accurate and confident, clearly identifying the article as POSITIVE with a 99.63% confidence score, but the summary was shorter and less detailed, covering only some of the article's key points rather than the full content.

## Upgrading to Cloud LLM Summarization with OpenRouter

### Integrating the OpenRouter free API

In this step, I'm upgrading the summarization feature by integrating OpenRouter and a cloud-based LLM so that I can generate more accurate, coherent, and detailed summaries and compare them with the local summarization results.

### LLM vs. local model output quality

The LLM summary is more detailed, well-structured, and easier to read compared to the local summary, which was brief and focused on only a few sections of the article. The local summarizer mainly highlighted AI applications in healthcare, finance, and transportation, but it left out other important points such as natural language processing advancements, data privacy concerns, algorithmic bias, job displacement, and the need for responsible AI governance. In contrast, the LLM summary captured both the benefits and challenges of AI, connected the ideas smoothly, and presented the information in a concise yet complete way. This showed that cloud-based LLMs are much better at understanding context and generating high-quality summaries than smaller local models

## Deploying a Polished Gradio Web Interface

### Building the tabbed UI

In this step, I'm building a Gradio web interface for my Article Analyzer so that I can transform my command-line AI tools into a professional, user-friendly web application that anyone can access through a browser. The interface will provide separate tabs for sentiment analysis and article summarization, allowing users to easily paste text, run analyses, and view results without needing to interact with Python code or the terminal. Additionally, Gradio will generate a shareable public URL, making it possible for others to access and test my application online...

![Image](https://nextwork.ai/proud_blue_vibrant_quince/uploads/6f6b9ea8-b2ca-434c-98e0-79cf1a7c5135_m35vexsi)

### Local vs. public share links

The local URL is used for testing and running the application on my own machine, while the public share link allows anyone on the internet to access and interact with the application through their browser. The local URL, such as http://127.0.0.1:7860, only works on the device where the app is running. The public Gradio link, such as https://xxxxx.gradio.live, tunnels internet traffic to the local application and can be shared with others, although it expires after 72 hours.

## Full Analysis Mode: Combining Both AI Backends

### Orchestrating sentiment and summarization together

In this project extension, I created a function that combines the results from two different AI backends by calling both the local sentiment analysis model and the cloud-based OpenRouter summarization model on the same input text. The function first analyzes the article's sentiment and returns a sentiment label with a confidence score, then generates a concise summary using the LLM. Finally, it combines both outputs into a single formatted response, displaying the sentiment analysis results and the article summary together. This allows users to get a complete analysis of an article with a single click instead of running each feature separately.

## Reflections and Key Takeaways

### Tools and concepts learned

The key tools I used include Hugging Face Transformers for sentiment analysis, OpenRouter for accessing a cloud-based LLM for summarization, Gradio for building an interactive web interface, Python virtual environments for managing project dependencies, and the OpenRouter API for connecting my application to AI models. Key concepts I learnt include how to use pre-trained machine learning models for sentiment analysis, how to integrate cloud-based LLMs into Python applications, how to securely manage API keys using environment variables, how to create and manage virtual environments, and how to build and deploy a user-friendly web application with Gradio. I also learned the difference between classification tasks, such as sentiment analysis, and generative tasks, such as text summarization, as well as the benefits of combining local AI models with cloud AI services in a single application.

### Time and challenges

This project took me approximately 1 day to complete. The most challenging part was troubleshooting the development environment, including Python installation and package dependency issues, as well as configuring the OpenRouter API and Gradio application correctly. Once the environment was set up, I was able to build the sentiment analysis feature, integrate cloud-based summarization, and create a user-friendly Gradio interface with separate tabs and a combined Full Analysis feature. The project helped me gain hands-on experience with AI model integration, API usage, debugging, and web application development.

### Looking ahead

I did this project today to learn how to build an AI-powered application that combines local machine learning models and cloud-based large language models to analyze and summarize text through an interactive web interface.

Another skill I want to learn is how to deploy AI applications to the cloud so that they can be accessed by users anywhere and scaled for real-world use.

---

*Built with [NextWork](https://nextwork.ai) - [View this project](https://nextwork.ai/projects/6f6b9ea8-b2ca-434c-98e0-79cf1a7c5135)*
