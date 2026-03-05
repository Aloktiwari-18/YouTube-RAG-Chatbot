import streamlit as st
import re
import os
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings


# ---------------- ENV ---------------- #

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("Google API key not found. Add it in .env file.")
    st.stop()
# ---------------- LLM ---------------- #

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2
)


# ---------------- Embeddings ---------------- #

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ---------------- Prompt ---------------- #

prompt = PromptTemplate(
template="""
You are a helpful assistant that answers questions about a YouTube video.

Rules:
- Use ONLY the context below
- Reply in the SAME language style as the user
- If user asks summary → give complete summary
- Mention approximate timestamp if relevant

Context:
{context}

Question:
{question}

Answer clearly.
""",
input_variables=["context","question"]
)


# ---------------- UI ---------------- #

st.set_page_config(page_title="YouTube RAG Chatbot", page_icon="▶", layout="wide")

st.title("🎥 YouTube RAG Chatbot")


# ---------------- Session ---------------- #

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "video_id" not in st.session_state:
    st.session_state.video_id = None

if "chunks" not in st.session_state:
    st.session_state.chunks = []


# ---------------- Extract Video ID ---------------- #

def extract_youtube_id(url):

    pattern = r"(?:v=|youtu\.be/|embed/|shorts/)([^&?/]+)"

    match = re.search(pattern, url)

    return match.group(1) if match else None


# ---------------- Sidebar ---------------- #

with st.sidebar:

    st.header("Load YouTube Video")

    url = st.text_input("Paste YouTube URL")

    if st.button("Load Video"):

        video_id = extract_youtube_id(url)

        if video_id:

            try:

                with st.spinner("Fetching transcript..."):

                    api = YouTubeTranscriptApi()

                    transcript = api.fetch(video_id)

                    full_text = " ".join(
                        chunk.text for chunk in transcript.snippets
                    )

                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1500,
                        chunk_overlap=200
                    )

                    docs = splitter.create_documents([full_text])

                    st.session_state.chunks = docs

                    vector_store = FAISS.from_documents(
                        docs,
                        embeddings
                    )

                    st.session_state.vector_store = vector_store
                    st.session_state.video_id = video_id

                st.success("Video Loaded Successfully!")

            except TranscriptsDisabled:
                st.error("Captions disabled for this video.")

            except NoTranscriptFound:
                st.error("No transcript available.")

            except Exception:
                st.error("Error loading transcript.")

        else:
            st.error("Invalid URL")

    if st.session_state.video_id:

        st.video(
            f"https://www.youtube.com/watch?v={st.session_state.video_id}"
        )

    st.divider()

    if st.button("📄 Video Summary"):

        if st.session_state.vector_store:

            st.session_state.messages.append(
                {"role":"user","content":"Give complete summary of the video"}
            )

        else:
            st.warning("Load a video first")

    if st.button("🧹 Clear Chat"):

        st.session_state.messages = []
        st.rerun()


# ---------------- Chat History ---------------- #

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ---------------- Chat Input ---------------- #

question = st.chat_input("Ask something about the video")

if question:

    st.session_state.messages.append(
        {"role":"user","content":question}
    )


# ---------------- AI Response ---------------- #

if st.session_state.messages:

    last = st.session_state.messages[-1]

    if last["role"] == "user":

        if st.session_state.vector_store is None:

            st.error("Please load a video first.")

        else:

            question = last["content"]

            with st.chat_message("assistant"):

                placeholder = st.empty()

                response = ""

                with st.spinner("AI thinking..."):

                    retriever = st.session_state.vector_store.as_retriever(
                        search_kwargs={"k":6}
                    )

                    docs = retriever.invoke(question)

                    context = "\n\n".join(
                        d.page_content for d in docs
                    )

                    final_prompt = prompt.invoke({
                        "context":context,
                        "question":question
                    })

                    try:

                        for chunk in llm.stream(final_prompt):

                            if chunk.content:

                                response += chunk.content
                                placeholder.markdown(response)

                    except Exception:

                        response = "AI service unavailable."
                        placeholder.markdown(response)

                st.session_state.messages.append(
                    {"role":"assistant","content":response}
                )