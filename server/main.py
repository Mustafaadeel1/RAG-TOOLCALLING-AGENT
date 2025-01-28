import streamlit as st
import base64
from rag_pipeline import initialize_rag_pipeline
from agent import create_agent
from file_upload import handle_file_upload
from tools import speech_to_text, text_to_speech
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential

# Configuration
GOOGLE_API_KEY = "AIzaSyB-vfL2QxAjrP_DecMiIBLr39xSucgQytk"
MAX_FILE_SIZE_MB = 200  # 200MB limit

# Session State Initialization
if 'rag_chain' not in st.session_state:
    st.session_state.rag_chain = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'voice_input' not in st.session_state:
    st.session_state.voice_input = False
if 'process_query' not in st.session_state:
    st.session_state.process_query = False

@st.cache_resource
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def load_components():
    try:
        agent = create_agent(GOOGLE_API_KEY)
        return agent
    except Exception as e:
        st.error(f"Initialization error: {str(e)}")
        st.stop()

def set_custom_theme():
    st.markdown("""
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
        .stTextInput input {
            background: #333333 !important;
            color: white !important;
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
        .sidebar-header {
            color: #4CAF50 !important;
            font-size: 1.5rem;
            text-shadow: 1px 1px 2px #000;
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
    """, unsafe_allow_html=True)

def main():
    set_custom_theme()
    
    st.markdown("""
    <h1 style='text-align: center; 
                color: #4CAF50;
                font-family: "Arial Black", sans-serif;
                text-shadow: 2px 2px 4px #000000;
                margin-bottom: 30px;'>
        🚀 AI Quantum Nexus
    </h1>
    """, unsafe_allow_html=True)

    agent = load_components()

    # Advanced Sidebar
    with st.sidebar.expander("⚙️ System Configuration", expanded=True):
        response_mode = st.selectbox(
            "Response Mode",
            ["Precise", "Balanced", "Creative"],
            index=1,
            help="Adjust the AI's creativity level"
        )
        
        # Dynamic temperature mapping
        temp_map = {"Precise": 0.1, "Balanced": 0.3, "Creative": 0.7}
        current_temp = temp_map[response_mode]

    # AI Toolkit Section
    with st.sidebar.expander("🛠️ AI Toolkit", expanded=True):
        st.markdown("""
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
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="margin-top: 15px; color: #888; font-size: 0.9rem; line-height: 1.4;">
            <span style="color: #4CAF50;">Core Capabilities:</span><br>
            • Natural Language Understanding<br>
            • Multi-modal Interaction<br>
            • Context-aware Responses<br>
            • Secure Data Handling
        </div>
        """, unsafe_allow_html=True)

    # File Processing Section
    with st.sidebar.expander("📁 Knowledge Management", expanded=True):
        uploaded_file = st.file_uploader(
            "Upload Documents",
            type=["pdf", "txt", "md"],
            help="Add files for enhanced context-aware responses"
        )
        
        if uploaded_file:
            if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
                st.error(f"File size exceeds {MAX_FILE_SIZE_MB}MB limit")
            else:
                with st.status("🧠 Processing Knowledge Base...", state="running") as status:
                    try:
                        st.session_state.rag_chain = handle_file_upload(uploaded_file, GOOGLE_API_KEY)
                        status.update(label="✅ Knowledge Integrated", state="complete")
                    except Exception as e:
                        status.update(label="❌ Processing Failed", state="error")
                        st.error(f"Error processing file: {str(e)}")

        st.markdown("""
        <div style="margin-top: 20px; border-top: 1px solid #4CAF50; padding-top: 15px;">
            <p style="color: #888; font-size: 0.9rem;">
            🚀 Supported formats: PDF, TXT, MD<br>
            ⚡ Max file size: 200MB<br>
            🔒 Secure processing guaranteed
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Chat Interface
    col1, col2 = st.columns([5, 1])
    with col1:
        user_input = st.text_area(
            "💬 Your Query",
            height=150,
            placeholder="Ask anything or give instructions...",
            key="user_input"
        )
    
    with col2:
        if st.button("🚀 Submit", use_container_width=True):
            st.session_state.process_query = True
        if st.button("🎤 Voice Input", use_container_width=True):
            st.session_state.voice_input = True

    # Process Voice Input
    if st.session_state.voice_input:
        with st.spinner("🎤 Listening..."):
            try:
                user_input = speech_to_text.invoke({})
                if isinstance(user_input, dict) and 'error' in user_input:
                    raise Exception(user_input['error'])
                    
                st.session_state.user_input = user_input
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Voice input error: {str(e)}")
                st.error("Please ensure microphone access and try again")
        st.session_state.voice_input = False

    # Process Queries
    if st.session_state.process_query and user_input.strip():
        with st.status("🧠 Processing Query...", expanded=True) as status:
            try:
                if st.session_state.rag_chain:
                    response = st.session_state.rag_chain.invoke({"input": user_input})["answer"]
                else:
                    response = agent.run({
                        "input": user_input,
                        "temperature": current_temp
                    })
                
                st.session_state.chat_history.append({
                    "query": user_input,
                    "response": response
                })
                
                with st.container():
                    st.subheader("💡 AI Response")
                    st.markdown(f"""
                    <div style="background: #333333;
                                padding: 20px;
                                border-radius: 12px;
                                margin: 10px 0;
                                box-shadow: 0 4px 8px rgba(0,0,0,0.2);">
                        {response}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    with st.expander("📊 Deep Analysis"):
                        deep_response = agent.run(f"Explain in detail: {response}")
                        st.markdown(f"```\n{deep_response}\n```")
                    
                    if text_to_speech:
                        audio_file = text_to_speech.invoke({"text": response})
                        st.audio(audio_file, format="audio/wav")

            except Exception as e:
                st.error(f"🚨 Error processing query: {str(e)}")
            
            status.update(label="✅ Processing Complete", state="complete")
        st.session_state.process_query = False

if __name__ == "__main__":
    main()