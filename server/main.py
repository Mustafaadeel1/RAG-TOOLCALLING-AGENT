import streamlit as st
from rag_pipeline import initialize_rag_pipeline
from agent import create_agent
from file_upload import handle_file_upload
from tools import speech_to_text, text_to_speech
from tenacity import retry, stop_after_attempt, wait_exponential

# Configuration
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
MAX_FILE_SIZE_MB = 200

# Session State Initialization
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "voice_input" not in st.session_state:
    st.session_state.voice_input = False
if "process_query" not in st.session_state:
    st.session_state.process_query = False

# Helper function moved to top
def get_response_text(response):
    """Extract text from various response formats"""
    if isinstance(response, dict):
        return response.get('answer', 
                   response.get('output', 
                       response.get('result', 
                           response.get('text', str(response)))))
    return str(response)

@st.cache_resource
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def load_components():
    try:
        return create_agent(GOOGLE_API_KEY)
    except Exception as e:
        st.error(f"Initialization error: {str(e)}")
        st.stop()

def set_custom_theme():
    st.markdown(
        """
    <style>
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(45deg, #1a1a1a, #2a2a2a);
            color: #ffffff;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(145deg, #1e1e1e, #2a2a2a) !important;
            border-right: 2px solid #4CAF50 !important;
            box-shadow: 5px 0 15px rgba(0,0,0,0.3) !important;
        }
        .stTextInput input, .stTextArea textarea {
            background: #333333 !important;
            color: white !important;
            border-radius: 8px;
        }
        .stSelectbox div {
            background: #333333 !important;
        }
        .stButton>button {
            background: #4CAF50;
            color: white;
            border-radius: 8px;
            padding: 12px 24px;
            transition: all 0.3s;
        }
        .stButton>button:hover {
            background: #45a049;
            transform: scale(1.05);
        }
        .tool-card {
            padding: 10px;
            background: #333;
            border-radius: 8px;
            text-align: center;
            transition: transform 0.2s;
        }
        .tool-card:hover {
            transform: scale(1.05);
        }
    </style>
    """,
        unsafe_allow_html=True,
    )

def main():
    set_custom_theme()

    st.markdown(
        """
    <h1 style='text-align: center; color: #4CAF50; 
                font-family: "Arial Black", sans-serif;
                text-shadow: 2px 2px 4px #000000;
                margin-bottom: 30px;'>
        🚀 AI Quantum Nexus
    </h1>
    """,
        unsafe_allow_html=True,
    )

    agent = load_components()

    # Sidebar Configuration
    with st.sidebar:
        with st.expander("⚙️ System Configuration", expanded=True):
            response_mode = st.selectbox(
                "Response Mode",
                ["Precise", "Balanced", "Creative"],
                index=1,
                help="Adjust the AI's creativity level",
            )
            temp_map = {"Precise": 0.1, "Balanced": 0.3, "Creative": 0.7}
            current_temp = temp_map[response_mode]

        # AI Toolkit Section
        with st.expander("🛠️ AI Toolkit", expanded=True):
            st.markdown(
                """
            <div style="border-radius: 10px; padding: 15px; background: #2a2a2a;">
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px;">
                    <div class="tool-card">
                        🎤<br>Speech Recognition
                    </div>
                    <div class="tool-card">
                        🔊<br>Text-to-Speech
                    </div>
                    <div class="tool-card">
                        📚<br>Document Analysis
                    </div>
                    <div class="tool-card">
                        🧠<br>Contextual Reasoning
                    </div>
                    <div class="tool-card">
                        🔍<br>Semantic Search
                    </div>
                    <div class="tool-card">
                        ⚡<br>Real-time Processing
                    </div>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with st.expander("📁 Knowledge Management", expanded=True):
            uploaded_file = st.file_uploader(
                "Upload Documents",
                type=["pdf", "txt", "md"],
                help="Max file size: 200MB",
            )
            st.markdown(
                """
            <div style="margin-top: 20px; border-top: 1px solid #4CAF50; padding-top: 15px;">
                <p style="color: #888; font-size: 0.9rem;">
                🚀 Supported formats: PDF, TXT, MD<br>
                ⚡ Max file size: 200MB<br>
                🔒 Secure processing guaranteed
                </p>
            </div>
            """,
                unsafe_allow_html=True,
            )

    # File Processing
    if uploaded_file and uploaded_file.size <= MAX_FILE_SIZE_MB * 1024 * 1024:
        with st.status("🧠 Processing Knowledge Base...", state="running") as status:
            try:
                st.session_state.rag_chain = handle_file_upload(uploaded_file, GOOGLE_API_KEY)
                status.update(label="✅ Knowledge Integrated", state="complete")
            except Exception as e:
                status.update(label="❌ Processing Failed", state="error")
                st.error(f"Error processing file: {str(e)}")
    elif uploaded_file:
        st.error(f"File size exceeds {MAX_FILE_SIZE_MB}MB limit")

    # Chat Interface
    col1, col2 = st.columns([5, 1])
    with col1:
        user_input = st.text_area(
            "💬 Your Query",
            height=150,
            placeholder="Ask anything or give instructions...",
            key="user_input",
        )

    with col2:
        if st.button("🚀 Submit", use_container_width=True):
            st.session_state.process_query = True
        if st.button("🎤 Voice Input", use_container_width=True):
            st.session_state.voice_input = True

    # Voice Input Handling
    if st.session_state.voice_input:
        with st.spinner("🎤 Listening..."):
            try:
                user_input = speech_to_text.invoke({})
                if isinstance(user_input, dict) and "error" in user_input:
                    raise Exception(user_input["error"])
                st.session_state.user_input = user_input
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Voice input error: {str(e)}")
                st.error("Please ensure microphone access and try again")
        st.session_state.voice_input = False

    # Query Processing - Corrected Section
    if st.session_state.process_query and isinstance(user_input, str) and user_input.strip():
        processing_container = st.empty()
        
        with processing_container.status("🧠 Processing Query...", expanded=True) as status:
            try:
                final_response = ""
                max_retries = 2
                
                if st.session_state.rag_chain:
                    for attempt in range(max_retries):
                        try:
                            response_obj = st.session_state.rag_chain.invoke({"input": user_input})
                            final_response = get_response_text(response_obj)
                            if final_response.strip() in ["", "No answer found"]:
                                raise ValueError("Empty RAG response")
                            break
                        except Exception as e:
                            if attempt == max_retries - 1:
                                st.warning("⚠️ RAG system struggling, using AI agent...")
                                final_response = get_response_text(
                                    agent.run({
                                        "input": f"Please answer: {user_input}",
                                        "temperature": current_temp
                                    })
                                )
                            continue
                else:
                    final_response = get_response_text(
                        agent.run({
                            "input": user_input,
                            "temperature": current_temp
                        })
                    )

                # Final validation
                final_response = str(final_response).strip()
                if not final_response:
                    final_response = "🔍 Sorry, I couldn't find a good answer. Try rephrasing your question."

            except Exception as e:
                final_response = f"🚨 Critical processing error: {str(e)}"
                st.error("Please try again or rephrase your question")

        # Clear processing container before showing results
        processing_container.empty()

        # Store response in session state after processing
        st.session_state.chat_history.append({
            "query": user_input,
            "response": final_response
        })

        # Display Response OUTSIDE the status block
        with st.container():
            st.subheader("💡 AI Response")
            response_box = st.markdown(
                f"""
                <div style="background: #333333;
                            padding: 20px;
                            border-radius: 12px;
                            margin: 10px 0;
                            box-shadow: 0 4px 8px rgba(0,0,0,0.2);">
                    {final_response}
                </div>
                """,
                unsafe_allow_html=True
            )

if __name__ == "__main__":
    main()
