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
        Tool.from_function(
            func=lambda text: text_to_speech.invoke({"text": text}),
            name="Text2Speech",
            description="Converts text to speech. Input should be the text to convert."
        ),
        Tool.from_function(
            func=lambda: speech_to_text.invoke({}),
            name="Speech2Text",
            description="Converts speech to text from microphone input. No input required."
        ),
        Tool.from_function(
            func=lambda image_path: image_to_text.invoke({"image_path": image_path}),
            name="Image2Text",
            description="Extracts text from images. Input should be the image file path."
        ),
        Tool.from_function(
            func=lambda expr: Calculator.invoke({"expression": expr}),
            name="Calculator",
            description="Solves math expressions starting with '='. Example input: '=2+2'"
        ),
        Tool.from_function(
            func=lambda _: fetch_latest_news.invoke({}),
            name="NewsFetcher",
            description="Fetches latest headlines. No input required."
        ),
        Tool.from_function(
            func=lambda _: get_random_joke.invoke({}),
            name="JokeGenerator",
            description="Tells random jokes. No input required."
        ),
        Tool.from_function(
            func=lambda ticker: fetch_stock_data.invoke({"ticker": ticker}),
            name="StockAnalyzer",
            description="Gets stock data. Input should be a ticker symbol like 'AAPL'"
        ),
        Tool.from_function(
            func=lambda query: search_everything.invoke({"query": query}),
            name="WebSearch",
            description="Searches the web. Input should be search query."
        )
    ]

    return initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors="Check your output and try again!",
        max_iterations=5,
        early_stopping_method="generate"
    )