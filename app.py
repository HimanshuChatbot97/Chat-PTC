import streamlit as st
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import os
import glob
import time

st.set_page_config(page_title="Document QA Chatbot")
st.title("📚 Ask Questions from Your Documents")

# Step 1: API key input
openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")
if not openai_api_key:
    st.stop()

os.environ["OPENAI_API_KEY"] = openai_api_key

# Step 2: Load documents from /docs folder
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

# Step 3: Retry wrapper for embeddings to avoid rate limits
def embed_documents_with_retry(embedding, texts, batch_size=10, max_retries=5):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        retries = 0
        while retries <= max_retries:
            try:
                emb = embedding.embed_documents(batch)
                all_embeddings.extend(emb)
                break
            except Exception as e:
                wait_time = 2 ** retries
                st.warning(f"Rate limit or API error. Retrying in {wait_time}s... ({e})")
                time.sleep(wait_time)
                retries += 1
        else:
            st.error("Embedding failed after multiple retries.")
            raise RuntimeError("Embedding failed.")
    return all_embeddings

# Step 4: Create or load FAISS vectorstore
def create_or_load_vectorstore(docs, embeddings):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)

    texts = [doc.page_content for doc in split_docs]
    metadatas = [doc.metadata for doc in split_docs]
    vectors = embed_documents_with_retry(embeddings, texts)
    vectordb = FAISS.from_embeddings(vectors, texts, metadatas=metadatas)
    return vectordb

# Step 5: Run everything
with st.spinner("Loading documents and preparing vectorstore..."):
    documents = load_documents()
    if not documents:
        st.error("No valid PDF or TXT documents found in the `docs` folder.")
        st.stop()

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectordb = create_or_load_vectorstore(documents, embeddings)

    retriever = vectordb.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=ChatOpenAI(api_key=openai_api_key), retriever=retriever)

st.success("✅ Documents loaded and ready!")

# Step 6: Ask questions
query = st.text_input("Ask a question:")
if query:
    result = qa_chain.run(query)
    st.write("🤖", result)
