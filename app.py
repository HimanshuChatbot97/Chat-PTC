import streamlit as st
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import os
import glob
import pickle

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

def create_or_load_vectorstore():
    if os.path.exists("faiss_index.pkl") and os.path.exists("faiss_index.pkl.metadatas"):
        with open("faiss_index.pkl", "rb") as f:
            vectordb = pickle.load(f)
        return vectordb
    else:
        documents = load_documents()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        split_docs = splitter.split_documents(documents)

        embeddings = OpenAIEmbeddings()
        vectordb = FAISS.from_documents(split_docs, embeddings)

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
