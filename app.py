import os
import streamlit as st
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import glob

st.set_page_config(page_title="Document QA Chatbot")
st.title("📚 Ask Questions")

openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")
if not openai_api_key:
    st.stop()
os.environ["OPENAI_API_KEY"] = openai_api_key

def load_documents():
    docs = []
    for file_path in glob.glob("docs/*"):
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith(".txt"):
            loader = TextLoader(file_path)
        else:
            continue
        docs.extend(loader.load())
    return docs

index_folder = "faiss_index"
embeddings = OpenAIEmbeddings()

if os.path.exists(index_folder):
    vectordb = FAISS.load_local(index_folder, embeddings)
else:
    st.warning("No index found. Please build the FAISS index first.")
    if st.button("Build FAISS Index"):
        with st.spinner("Building index. This might take a few minutes..."):
            documents = load_documents()
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            split_docs = splitter.split_documents(documents)
            vectordb = FAISS.from_documents(split_docs, embeddings)
            vectordb.save_local(index_folder)
            st.success("Index built! Please rerun the app.")
        st.stop()
    else:
        st.stop()

retriever = vectordb.as_retriever()
qa_chain = RetrievalQA.from_chain_type(llm=ChatOpenAI(), retriever=retriever)

query = st.text_input("Ask a question:")
if query:
    result = qa_chain.run(query)
    st.write("🤖", result)
