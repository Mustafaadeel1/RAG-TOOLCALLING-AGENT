import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def handle_file_upload(google_api_key):
    st.sidebar.title("Upload Your File")
    uploaded_file = st.sidebar.file_uploader("Upload a file (PDF or text)", type=["pdf", "txt"])

    if uploaded_file is not None:
        with open("temp_file", "wb") as f:
            f.write(uploaded_file.getbuffer())

        if uploaded_file.type == "application/pdf":
            loader = PyPDFLoader("temp_file")
        else:
            loader = TextLoader("temp_file")

        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
        docs = text_splitter.split_documents(data)

        embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
        vectorstore = FAISS.from_documents(docs, embedding)
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

        return {"retriever": retriever}
    return None
