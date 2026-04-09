import streamlit as st
import re
import os
import time
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from youtube_transcript_api import YouTubeTranscriptApi

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings


# ---------------- ENV ---------------- #

load_dotenv()

try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


# ---------------- LLM ---------------- #

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY
)


# ---------------- Embeddings ---------------- #

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)


# ---------------- Prompt ---------------- #

prompt = PromptTemplate(
template="""
You are an intelligent AI assistant that answers questions about a YouTube video using the provided transcript context.

STRICT RULES:
1. Use ONLY the provided context to answer the question.
2. If the answer is not present in the context, say:
   "The information is not available in the video transcript."
3. Do NOT hallucinate or make up information.
4. Keep answers accurate and grounded in the context.

LANGUAGE RULE:
Respond in the SAME language as the user's question.

Context:
{context}

User Question:
{question}

Answer:
""",
input_variables=["context", "question"]
)


# ---------------- Streamlit ---------------- #

st.set_page_config(page_title="YouTube RAG Chatbot", page_icon="▶")
st.title("🎥 YouTube RAG Chatbot")


# ---------------- Session ---------------- #

if "messages" not in st.session_state:
    st.session_state.messages = []

if "video_id" not in st.session_state:
    st.session_state.video_id = None

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None


# ---------------- Extract Video ID ---------------- #

def extract_youtube_id(url):
    pattern = r"(?:v=|youtu\.be/|embed/|shorts/)([^&?/]+)"
    match = re.search(pattern, url)
    return match.group(1) if match else None


# ---------------- Transcript Loader ---------------- #

@st.cache_data(show_spinner=False)
def load_transcript(video_id):

    api = YouTubeTranscriptApi()

    for i in range(3):  # retry logic
        try:
            transcript = api.get_transcript(
                video_id,
                languages=["en", "hi"]
            )

            text = " ".join(chunk["text"] for chunk in transcript)

            if text.strip():
                return text

        except Exception as e:
            print("Retry:", i, "Error:", e)
            time.sleep(2)

    return ""


# ---------------- Sidebar ---------------- #

with st.sidebar:

    st.header("Load YouTube Video")

    url = st.text_input("Paste YouTube URL")

    if st.button("Load Video"):

        video_id = extract_youtube_id(url)

        if video_id:

            st.session_state.video_id = video_id

            try:

                with st.spinner("Fetching transcript..."):

                    text = load_transcript(video_id)

                    # ✅ FIX 1: Empty transcript check
                    if not text or len(text.strip()) == 0:
                        st.error("❌ Transcript not available or blocked by YouTube.")
                        st.stop()

                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1500,
                        chunk_overlap=300
                    )

                    docs = splitter.create_documents([text])

                    # ✅ FIX 2: Documents check
                    if not docs or len(docs) == 0:
                        st.error("❌ No content to process.")
                        st.stop()

                    vector_store = FAISS.from_documents(
                        docs,
                        embeddings
                    )

                    st.session_state.vector_store = vector_store

                st.success("✅ Video Loaded and Indexed!")

            except Exception as e:
                st.error("❌ Something went wrong.")
                st.write(e)

        else:
            st.error("Invalid URL")

    if st.session_state.video_id:
        st.video(
            f"https://www.youtube.com/watch?v={st.session_state.video_id}"
        )

# ---------------- Chat History ---------------- #

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ---------------- Chat ---------------- #

question = st.chat_input("Ask something about the video")

if question:

    if st.session_state.vector_store is None:
        st.error("Please load a video first.")
        st.stop()

    st.session_state.messages.append(
        {"role":"user","content":question}
    )

    with st.chat_message("user"):
        st.markdown(question)


    retriever = st.session_state.vector_store.as_retriever(
        search_kwargs={"k":8}
    )

    docs = retriever.invoke(question)

    context = "\n\n".join(
        d.page_content for d in docs
    )

    final_prompt = prompt.invoke({
        "context":context,
        "question":question
    })


    with st.chat_message("assistant"):

        response = ""
        placeholder = st.empty()

        for chunk in llm.stream(final_prompt):
            response += chunk.content
            placeholder.markdown(response)

    st.session_state.messages.append(
        {"role":"assistant","content":response}
    )
