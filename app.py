import streamlit as st
import os
import glob
import pickle
import time
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

st.set_page_config(page_title="Document QA Chatbot")
st.title("📚 Ask Questions")

openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")
if not openai_api_key:
    st.stop()
os.environ["OPENAI_API_KEY"] = openai_api_key

def load_documents():
    docs = []
    files = glob.glob("docs/*")
    for file_path in files:
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith(".txt"):
            loader = TextLoader(file_path)
        else:
            continue
        docs.extend(loader.load())
    return docs

def embed_documents_with_retry(embedding, texts, batch_size=10, max_retries=5):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        retrie
