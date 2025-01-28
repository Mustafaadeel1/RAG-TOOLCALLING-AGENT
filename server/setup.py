import os
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

def setup_faiss():
    GOOGLE_API_KEY = "AIzaSyBovIbJE46S_3hvSb4wLdZvGPpYsSWPVow"
    PDF_PATH = "C:/Users/USER/Downloads/typescript-handbook.pdf"

    if os.path.exists("faiss_index"):
        print("♻️ Deleting existing FAISS index...")
        shutil.rmtree("faiss_index")

    print("🔄 Loading and splitting PDF...")
    loader = PyPDFLoader(PDF_PATH)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
    docs = text_splitter.split_documents(data)

    print("🔧 Creating embeddings...")
    embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

    print("💾 Saving FAISS index...")
    vectorstore = FAISS.from_documents(docs, embedding)
    vectorstore.save_local("faiss_index")
    print("✅ Setup completed successfully!")

if __name__ == "__main__":
    setup_faiss()