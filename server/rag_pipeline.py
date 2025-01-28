import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from typing import Optional
import logging

logger = logging.getLogger(__name__)

def initialize_rag_pipeline(google_api_key: str, retriever=None):
    """
    Initialize RAG pipeline with enhanced safety and validation
    
    Args:
        google_api_key: Valid Google API key
        retriever: Optional pre-configured retriever
    """
    # Validate API key format
    if not google_api_key.startswith("AIza"):
        raise ValueError("Invalid Google API key format")

    # Initialize embeddings with connection check
    try:
        embedding = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", 
            google_api_key=google_api_key
        )
        # Test embedding connection
        embedding.embed_query("test")
    except Exception as e:
        logger.error("Embedding initialization failed", exc_info=True)
        raise RuntimeError("Failed to initialize embeddings") from e

    # Retriever handling
    if retriever is None:
        if not os.path.exists("faiss_index"):
            raise FileNotFoundError("FAISS index directory not found")
            
        if not os.listdir("faiss_index"):
            raise ValueError("FAISS index directory is empty")

        try:
            vectorstore = FAISS.load_local(
                folder_path="faiss_index",
                embeddings=embedding,
                allow_dangerous_deserialization=True
            )
            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 10}
            )
        except Exception as e:
            logger.error("FAISS index loading failed", exc_info=True)
            raise RuntimeError("Failed to load vector store") from e

    # LLM initialization with validation
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            temperature=0.3,
            max_tokens=1000,
            google_api_key=google_api_key
        )
        # Test LLM connection
        llm.invoke("test")
    except Exception as e:
        logger.error("LLM initialization failed", exc_info=True)
        raise RuntimeError("Failed to initialize language model") from e

    # Enhanced prompt template
    system_prompt = """You are an expert QA assistant. Follow these rules:
    1. Use ONLY the provided context
    2. If unsure, say "I don't have enough information"
    3. Keep answers under 3 sentences
    4. Never hallucinate information
    5. Format responses clearly using simple markdown
    
    Context: {context}"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # Create chains with validation
    try:
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        return create_retrieval_chain(retriever, question_answer_chain)
    except Exception as e:
        logger.error("Chain creation failed", exc_info=True)
        raise RuntimeError("Failed to create processing chain") from e