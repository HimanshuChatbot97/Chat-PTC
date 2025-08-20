import os
import glob
import time
import streamlit as st
from typing import List
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from openai import OpenAIError

# Streamlit UI
st.set_page_config(page_title="Document QA Chatbot")
st.title("📚 Ask Questions")

openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")
if not openai_api_key:
    st.stop()

os.environ["OPENAI_API_KEY"] = openai_api_key

# Configs
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
BATCH_SIZE = 5
RETRY_DELAY = 5  # seconds

# Load documents
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

# Embed documents in small batches with retry on 429
def embed_in_batches(docs, embeddings, batch_size=5):
    texts = [doc.page_content for doc in docs]
    metadatas = [doc.metadata for doc in docs]
    embedded_batches = []
    metadata_batches = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_metadata = metadatas[i:i + batch_size]
        success = False
        retries = 0
        while not success:
            try:
                vectors = embeddings.embed_documents(batch)
                embedded_batches.extend(vectors)
                metadata_batches.extend(batch_metadata)
                success = True
            except OpenAIError as e:
                st.warning(f"Rate limit or API error. Retrying in {RETRY_DELAY}s... ({str(e)})")
                time.sleep(RETRY_DELAY)
                retries += 1
                if retries > 5:
                    st.error("Too many retries. Exiting.")
                    st.stop()
    return embedded_batches, metadata_batches, texts

# Vector store creation
def create_vectorstore(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    split_docs = text_splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings()

    st.info(f"Splitting into {len(split_docs)} chunks. Embedding in batches of {BATCH_SIZE}...")

    vectors, metadata, raw_texts = embed_in_batches(split_docs, embeddings, BATCH_SIZE)
    vectordb = FAISS.from_embeddings(vectors, raw_texts, metadata)
    return vectordb

# Main logic
with st.spinner("Loading and embedding documents..."):
    documents = load_documents()
    vectordb = create_vectorstore(documents)
    retriever = vectordb.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=ChatOpenAI(), retriever=retriever)

st.success("✅ Ready to chat with your documents!")

# Ask question
query = st.text_input("Ask a question:")
if query:
    with st.spinner("Thinking..."):
        result = qa_chain.run(query)
        st.write("🤖", result)
