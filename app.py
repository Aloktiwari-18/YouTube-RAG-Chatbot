import streamlit as st
import re
import os
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
You are a helpful assistant that answers questions about a YouTube video.

Rules:
Use ONLY the context below.

Reply in the SAME language style as the user.
English → English
Hinglish → Hinglish (English letters)

Context:
{context}

Question:
{question}

Answer clearly.
""",
input_variables=["context","question"]
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

    transcript = api.fetch(
        video_id,
        languages=["en", "hi"]
    )

    text = " ".join(chunk.text for chunk in transcript.snippets)

    return text


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

                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1500,
                        chunk_overlap=300
                    )

                    docs = splitter.create_documents([text])

                    vector_store = FAISS.from_documents(
                        docs,
                        embeddings
                    )

                    st.session_state.vector_store = vector_store

                st.success("Video Loaded and Indexed!")

            except Exception as e:

                st.error("Transcript not available for this video.")
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