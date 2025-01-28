import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS

def initialize_rag_pipeline(google_api_key, retriever=None):
    """
    Initialize RAG pipeline with FAISS vector store.

    Args:
        google_api_key (str): Google API key for embeddings.
        retriever (Optional): Pre-configured retriever. If None, a default retriever is created.
    """
    # Load embeddings
    embedding = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", 
        google_api_key=google_api_key
    )

    # Use the provided retriever or create a new one
    if retriever is None:
        # Security check
        if not os.path.exists("faiss_index"):
            raise FileNotFoundError("FAISS index not found. Run setup.py first!")

        # Load default FAISS index
        vectorstore = FAISS.load_local(
            folder_path="faiss_index",
            embeddings=embedding,
            allow_dangerous_deserialization=True
        )

        # Initialize retriever
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 10}
        )

    # Initialize LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        temperature=0.3,
        max_tokens=1000,
        google_api_key=google_api_key
    )

    # Define the system prompt
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the retrieved context to answer. If unsure, say you don't know. "
        "Keep answers concise (3 sentences max).\n\n{context}"
    )

    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # Create chains
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, question_answer_chain)