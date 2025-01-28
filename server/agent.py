from langchain.agents import initialize_agent, AgentType
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import Tool
from tools import (
    text_to_speech,
    speech_to_text,
    image_to_text,
    Calculator,
    fetch_latest_news,
    get_random_joke,
    fetch_stock_data,
    search_everything
)

def create_agent(google_api_key):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        temperature=0.3,
        max_tokens=1000,
        google_api_key=google_api_key
    )

    tools = [
        Tool(
            name="Text2Speech",
            func=text_to_speech.invoke,
            description="Converts text to speech"
        ),
        Tool(
            name="Speech2Text",
            func=speech_to_text.invoke,
            description="Converts speech to text"
        ),
        Tool(
            name="Image2Text",
            func=image_to_text.invoke,
            description="Extracts text from images"
        ),
        Tool(
            name="Calculator",
            func=Calculator.invoke,
            description="Performs mathematical calculations"
        ),
        Tool(
            name="News",
            func=fetch_latest_news.invoke,
            description="Fetches latest news"
        ),
        Tool(
            name="Jokes",
            func=get_random_joke.invoke,
            description="Tells random jokes"
        ),
        Tool(
            name="Stocks",
            func=fetch_stock_data.invoke,
            description="Fetches stock market data"
        ),
        Tool(
            name="WebSearch",
            func=search_everything.invoke,
            description="Performs web searches"
        )
    ]

    return initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors="Check output formatting! Respond with valid JSON!",
        max_iterations=10
    )