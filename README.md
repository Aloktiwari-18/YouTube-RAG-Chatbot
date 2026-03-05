# 🎥 YouTube RAG Chatbot

An AI-powered chatbot that can answer questions about any **YouTube video** using **Retrieval Augmented Generation (RAG)**.

Users can paste a YouTube video link and ask questions about the video. The chatbot understands the transcript and generates intelligent answers using Large Language Models.

---

# 🚀 Features

* 🎥 Ask questions about any YouTube video
* 🧠 RAG based question answering
* 📜 Automatic transcript extraction
* ⚡ Fast semantic search using FAISS
* 🤖 Works with **Gemini API** (Cloud LLM)
* 💻 Works with **Ollama Local LLM**
* 💬 Supports **English and Hinglish conversations**
* 🌐 Deployable with Streamlit Cloud

---

# 🧠 How It Works

1️⃣ User pastes a **YouTube video URL**

2️⃣ The system extracts the **video transcript**

3️⃣ Transcript is split into smaller chunks

4️⃣ Chunks are converted into **vector embeddings**

5️⃣ Stored inside a **FAISS vector database**

6️⃣ When a user asks a question:

* Relevant transcript chunks are retrieved
* Context is sent to the LLM
* The model generates the final answer

---

# 🏗️ Architecture

```
User Question
      │
      ▼
YouTube Transcript API
      │
      ▼
Text Chunking
      │
      ▼
Embedding Model
      │
      ▼
FAISS Vector Database
      │
      ▼
Retriever
      │
      ▼
LLM (Gemini / Ollama)
      │
      ▼
Final Answer
```

---

# 🤖 LLM Support

This project supports **two different LLM modes**.

## 1️⃣ Gemini API (Cloud)

Used for the **deployed version**.

Advantages:

* Fast responses
* High accuracy
* Easy deployment
* No local GPU required

---

## 2️⃣ Ollama Local LLM

Used for **running the chatbot locally** without any API.

Models used:

* `llama3`
* `nomic-embed-text`

Advantages:

* Works completely offline
* No API cost
* Full control over the model
* Privacy friendly

---

# 🛠️ Tech Stack

* **Python**
* **Streamlit**
* **LangChain**
* **Google Gemini API**
* **Ollama**
* **FAISS Vector Database**
* **YouTube Transcript API**
* **Sentence Transformers**

---

# 📂 Project Structure

```
youtube-rag-chatbot
│
├── app.py                 # Gemini RAG chatbot (deploy version)
├── local_ollama_app.py    # Local Ollama RAG chatbot
├── requirements.txt
├── README.md
└── .gitignore
```

---

# ⚙️ Installation (Local Setup)

Clone the repository

```
git clone https://github.com/yourusername/youtube-rag-chatbot.git
```

Go to the project directory

```
cd youtube-rag-chatbot
```

Install dependencies

```
pip install -r requirements.txt
```

Run the app

```
streamlit run app.py
```

---

# 🔑 Environment Variables

Create a `.env` file:

```
GOOGLE_API_KEY=your_gemini_api_key
```

---

# 🌍 Deployment

This project can be easily deployed using **Streamlit Cloud**.

Steps:

1️⃣ Push the project to GitHub

2️⃣ Open **Streamlit Cloud**

3️⃣ Connect your GitHub repository

4️⃣ Select the main file:

```
app.py
```

5️⃣ Add the secret:

```
GOOGLE_API_KEY = your_api_key
```

6️⃣ Deploy 🚀

---

# 💡 Example Questions

Users can ask:

* "Give me the summary of this video"
* "Explain the Dynamic Programming concept"
* "What problem is solved in this video?"
* "Explain the algorithm used"

---

# 📸 Example Use Cases

* Summarizing long lectures
* Understanding coding tutorials
* Extracting key concepts from educational videos
* Learning programming topics faster

---

# 👨‍💻 Author

**Alok Tiwari**

BTech CSE Student
Interested in **AI, Data Science, and Web Development**

---

# ⭐ If you like this project

Please consider **starring the repository** ⭐

It helps others discover the project.
