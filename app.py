import os
import glob
import streamlit as st
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import HuggingFaceHub

st.title("📚 Document Q&A (Run Locally on Spaces)")

def load_docs():
    docs = []
    for fp in glob.glob("docs/*"):
        if fp.endswith(".pdf"):
            docs.extend(PyPDFLoader(fp).load())
        elif fp.endswith(".txt"):
            docs.extend(TextLoader(fp).load())
    return docs

with st.spinner("Loading documents..."):
    docs = load_docs()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(split_docs, embeddings)
    retriever = vectordb.as_retriever()

    llm = HuggingFaceHub(repo_id="google/flan-t5-small", model_kwargs={"temperature": 0, "max_length": 256})
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

st.success("✅ Ready to answer your questions!")

query = st.text_input("Ask a question:")
if query:
    answer = qa_chain.run(query)
    st.write("Answer:", answer)
