from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import tempfile
import os
import logging

logger = logging.getLogger(__name__)

def validate_file_type(file_type: str) -> bool:
    """Validate supported file types"""
    return file_type in ["application/pdf", "text/plain"]

def handle_file_upload(uploaded_file, google_api_key: str):
    """Process uploaded files and create vector store"""
    if not validate_file_type(uploaded_file.type):
        raise ValueError(f"Unsupported file type: {uploaded_file.type}")
    
    if uploaded_file.size > 100 * 1024 * 1024:  # 100MB limit
        raise ValueError("File size exceeds maximum allowed limit (100MB)")

    temp_path = None
    try:
        # Create temporary file with context manager
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            temp_path = tmp_file.name

        # Load document with appropriate loader
        if uploaded_file.type == "application/pdf":
            loader = PyPDFLoader(temp_path)
        else:
            loader = TextLoader(temp_path)
            
        documents = loader.load()

        # Split documents with validation
        if len(documents) == 0:
            raise ValueError("No readable content found in the document")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False
        )
        splits = text_splitter.split_documents(documents)

        # Create embeddings with connection validation
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=google_api_key
        )
        
        # Create vector store with progress tracking
        vectorstore = FAISS.from_documents(
            documents=splits,
            embedding=embeddings
        )

        return vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 10}
        )

    except Exception as e:
        logger.error(f"File processing error: {str(e)}", exc_info=True)
        raise RuntimeError(f"Document processing failed: {str(e)}") from e

    finally:
        # Cleanup temporary file
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except OSError as cleanup_error:
                logger.warning(f"Temp file cleanup failed: {str(cleanup_error)}")