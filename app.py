import streamlit as st
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from openai import OpenAIError
import os
import glob
import time

st.set_page_config(page_title="Document QA Chatbot")
st.title("📚 Ask Questions from your Docs")

openai_api_key = st.text_input("🔑 Enter your OpenAI API Key", type="password")
if not openai_api_key:
    st.stop()

os.environ["OPENAI_API_KEY"] = openai_api_key

@st.cache_data(show_spinner=False)
def load_documents(limit=3):
    docs = []
    files = glob.glob("docs/*")
    for i, file_path in enumerate(files):
        if i >= limit:  # To avoid exceeding token limits
            break
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith(".txt"):
            loader = TextLoader(file_path)
        else:
            continue
        docs.extend(loader.load())
    return docs

def create_embeddings_with_backoff(texts, embedding_model):
    """Embed texts with exponential backoff to avoid hitting rate limits."""
    results = []
    for i, text in enumerate(texts):
        retry = 0
        while retry < 5:
            try:
                result = embedding_model.embed_documents([text])
                results.extend(result)
                break
            except OpenAIError as e:
                if 'rate_limit' in str(e).lower() or '429' in str(e).lower():
                    wait_time = 2 ** retry
                    st.warning(f"⚠️ Rate limit hit. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    retry += 1
                else:
                    st.error(f"❌ Embedding failed: {e}")
                    break
    return results

with st.spinner("🧠 Processing documents..."):
    documents = load_documents(limit=3)
    if not documents:
        st.error("No valid documents found in `docs/` folder.")
        st.stop()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    split_docs = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    # Use retry mechanism
    try:
        vectordb = FAISS.from_documents(split_docs, embeddings)
    except OpenAIError as e:
        st.error(f"❌ Failed to create vectorstore due to OpenAI error: {e}")
        st.stop()

    retriever = vectordb.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=ChatOpenAI(openai_api_key=openai_api_key), retriever=retriever)

st.success("✅ Ready! Ask your questions below.")

query = st.text_input("💬 Ask a question:")
if query:
    with st.spinner("🤖 Thinking..."):
        try:
            result = qa_chain.run(query)
            st.write("🤖", result)
        except OpenAIError as e:
            st.error(f"OpenAI error during QA: {e}")
