# YouTube RAG Chatbot

A Retrieval-Augmented Generation chatbot that answers
questions from YouTube videos using transcript retrieval.

## Tech Stack

Python
LangChain
FAISS
Streamlit
Ollama (Llama3)
Google Gemini

## Run Locally

Install Ollama

ollama pull llama3
ollama pull nomic-embed-text

Install dependencies

pip install -r requirements.txt

Run

streamlit run app.py