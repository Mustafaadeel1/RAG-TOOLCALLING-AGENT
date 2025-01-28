from langchain_core.tools import tool
import math
import requests
from PIL import Image
import pyttsx3
import random
import speech_recognition as sr
import pytesseract
import json

# Set the path to the Tesseract-OCR executable (update this path if necessary)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

@tool
def text_to_speech(text: str):
    """
    Converts the given text to speech using the pyttsx3 library.

    Args:
        text (str): The text to convert to speech.

    Returns:
        str: Success message if the speech synthesis is completed.
        dict: Error message in case of failure.
    """
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
        return "Speech synthesis complete."
    except Exception as e:
        return {"error": str(e)}

@tool
def speech_to_text():
    """
    Converts speech input from the microphone to text using the SpeechRecognition library.

    Returns:
        str: The recognized text from the speech.
        dict: Error message in case of failure.
    """
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        try:
            print("Listening...")
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=5)
            text = recognizer.recognize_google(audio)
            return text
        except Exception as e:
            return {"error": str(e)}

@tool
def image_to_text(image_path: str) -> str:
    """
    Extracts text from the specified image using the pytesseract OCR library.

    Args:
        image_path (str): The path to the image file.

    Returns:
        str: The extracted text from the image.
        str: Error message if the image file is not found or another error occurs.
    """
    try:
        img = Image.open(image_path)
        extracted_text = pytesseract.image_to_string(img)
        return extracted_text.strip()
    except FileNotFoundError:
        return "\u274c Error: The specified image file was not found."
    except Exception as e:
        return f"\u274c Error: {str(e)}"

@tool
def Calculator(expression):
    """
    Evaluates a mathematical expression starting with '=' using safe evaluation.

    Args:
        expression (str): The mathematical expression to evaluate. Must start with '='.

    Returns:
        str: The result of the evaluated expression.
        str: Error message if the expression is invalid.
    """
    if not expression.startswith('='):
        return "Error: Expression must start with '='."
    expression = expression[1:].strip()
    allowed_functions = {
        'SUM': sum,
        'AVG': lambda *args: sum(args) / len(args) if args else 0,
        **{k: v for k, v in math.__dict__.items() if not k.startswith("__")}
    }
    try:
        result = eval(expression, {"__builtins__": None}, allowed_functions)
    except Exception as e:
        return f"Error in evaluation: {e}"
    return result

@tool
def fetch_latest_news(category='general', country='us'):
    """
    Fetches the latest news headlines from the NewsAPI.

    Args:
        category (str): The news category (default: 'general').
        country (str): The country code (default: 'us').

    Returns:
        str: The latest news headlines.
        dict: Error message in case of failure.
    """
    api_key = '39245a2fc2e24025a7644b5c6c1e2cc2'
    url = f"https://newsapi.org/v2/top-headlines?category={category}&country={country}&apiKey={api_key}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            articles = data.get('articles', [])
            if articles:
                news = [f"{i+1}. {article['title']} (Source: {article['source']['name']})" 
                        for i, article in enumerate(articles[:5])]
                return "\n".join(news)
            else:
                return "No news articles found."
        else:
            return f"Error: Unable to fetch news (Status Code: {response.status_code})"
    except Exception as e:
        return {"error": str(e)}

@tool
def get_random_joke():
    """
    Returns a random joke from a predefined list of jokes.

    Returns:
        str: A random joke.
        str: Error message in case of failure.
    """
    jokes = [
        "Why don't skeletons fight each other? They don't have the guts.",
        "What do you call fake spaghetti? An impasta.",
        "I told my wife she was drawing her eyebrows too high. She looked surprised.",
    ]
    try:
        return random.choice(jokes)
    except Exception as e:
        return f"Error occurred: {str(e)}"

@tool
def fetch_stock_data(ticker):
    """
    Fetches stock data for the specified ticker symbol from the Alpha Vantage API.

    Args:
        ticker (str): The stock ticker symbol.

    Returns:
        str: The stock data including current price, previous close, day's high, day's low, and volume.
        str: Error message if the stock data cannot be retrieved.
    """
    api_key = "AqWbGjogZ8vS2XDCQjo3X3A3495ouUGb"
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={ticker}&apikey={api_key}"
    try:
        response = requests.get(url)
        data = response.json()
        if "Global Quote" in data:
            quote = data["Global Quote"]
            return (
                f"Stock: {ticker.upper()}\n"
                f"Current Price: ${quote.get('05. price', 'N/A')}\n"
                f"Previous Close: ${quote.get('08. previous close', 'N/A')}\n"
                f"Day's High: ${quote.get('03. high', 'N/A')}\n"
                f"Day's Low: ${quote.get('04. low', 'N/A')}\n"
                f"Volume: {quote.get('06. volume', 'N/A')}\n"
            )
        else:
            return "Error: No stock data found for the given ticker."
    except Exception as e:
        return f"Error fetching stock data: {e}"

@tool
def search_everything(query):
    """
    Search Google using the Serper API and return structured results.

    Args:
    - query (str): The search query (e.g., 'latest AI news').

    Returns:
    - dict: A structured response containing the search results.
    """
    url = 'https://google.serper.dev/search'
    api_key = 'b65444ec6c261125db072707c300f0e274b9814b'
    headers = {
        'X-API-KEY': api_key,
        'Content-Type': 'application/json'
    }
    payload = {
        'q': query
    }

    try:
        # Make the API request
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        data = response.json()

        # Process and return the results
        results = data.get('organic', [])
        output = []
        for result in results:
            output.append({
                'Title': result.get('title'),
                'Link': result.get('link'),
                'Snippet': result.get('snippet')
            })
        return output

    except requests.exceptions.RequestException as e:
        return {"error": f"An error occurred: {e}"}