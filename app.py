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
from openai.error import OpenAIError

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
    for i in range(0, len(texts), batch_size):  # <-- fixed 'rang' -> 'range' and added ':'
        batch = texts[i : i + batch_size]
        retries = 0
        while True:
            try:
                emb = embedding.embed_documents(batch)
                all_embeddings.extend(emb)
                break
            except OpenAIError as e:
                if retries >= max_retries:
                    st.error("Max retries reached for embedding calls.")
                    raise
                wait_time = 2 ** retries
                st.warning(f"OpenAI API error, retrying in {wait_time} seconds... ({e})")
                time.sleep(wait_time)
                retries += 1
    return all_embeddings

def create_or_load_vectorstore():
    if os.path.exists("faiss_index.pkl"):
        with open("faiss_index.pkl", "rb") as f:
            vectordb = pickle.load(f)
        return vectordb
    else:
        documents = load_documents()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        split_docs = splitter.split_documents(documents)

        embeddings = OpenAIEmbeddings()

        texts = [doc.page_content for doc in split_docs]

        all_embs = embed_documents_with_retry(embeddings, texts)

        vectordb = FAISS.from_texts(texts, embeddings, metadatas=[doc.metadata for doc in split_docs])

        with open("faiss_index.pkl", "wb") as f:
            pickle.dump(vectordb, f)
        return vectordb

with st.spinner("Loading or creating vectorstore..."):
    vectordb = create_or_load_vectorstore()

retriever = vectordb.as_retriever()
qa_chain = RetrievalQA.from_chain_type(llm=ChatOpenAI(), retriever=retriever)

st.success("✅ Ready to answer questions!")

query = st.text_input("Ask a question:")
if query:
    with st.spinner("Thinking..."):
        result = qa_chain.run(query)
    st.write("🤖", result)
