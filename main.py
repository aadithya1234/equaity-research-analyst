import os
import streamlit as st
import pickle
import time
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load env variables
load_dotenv(override=True)
os.environ["LANGCHAIN_TRACING_V2"] = "false"

# Initialise Sarvam LLM
llm = ChatOpenAI(
    api_key="dummy",
    base_url="https://api.sarvam.ai/v1",
    model="sarvam-30b",
    temperature=0.9,
    max_tokens=500,
    default_headers={
        "api-subscription-key": os.getenv("SARVAM_API_KEY")
    }
)

# Streamlit UI
st.title("RockyBot: News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_sarvam.pkl"
main_placeholder = st.empty()

if process_url_clicked:
    # Load data using requests + BeautifulSoup (bypasses bot blocking)
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    data = []
    main_placeholder.text("Data Loading...Started...✅✅✅")
    for url in urls:
        if url.strip():  # skip empty URLs
            try:
                response = requests.get(url, headers=headers)
                soup = BeautifulSoup(response.text, "html.parser")
                text = soup.get_text(separator="\n", strip=True)
                data.append(Document(page_content=text, metadata={"source": url}))
            except Exception as e:
                st.warning(f"Could not load {url}: {e}")

    # Split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000,
        chunk_overlap=200
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)

    # Create embeddings using HuggingFace (free, no API key needed)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore_sarvam = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)

    # Save FAISS index
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_sarvam, f)
    main_placeholder.text("Index saved! Ready to answer questions ✅")

# Query section
query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)

        retriever = vectorstore.as_retriever()

        prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the following context.
Always mention the source URL at the end under 'Sources:'.

Context: {context}

Question: {question}
""")

        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        result = chain.invoke(query)

        st.header("Answer")
        st.write(result)
    else:
        st.warning("Please process URLs first using the sidebar.")