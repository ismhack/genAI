import os
from dotenv import load_dotenv, find_dotenv

from langchain import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain_community.tools import TavilySearchResults
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv(find_dotenv())

gemini_key = os.getenv("GEMINI_API_KEY")
tavily_key = os.getenv("TAVILY_API_KEY")
tavily_tool = TavilySearchResults(max_results=5)

from google import genai
client = genai.Client(api_key=gemini_key)
MODEL_ID="gemini-2.0-flash"

prompt = "Write a short essay in spanish about the history of mexico city rock and roll movement, include famous bands from the region and mention how the sorrounding neighbours like Nezahualcoyotl and Chimalhuacan influenced the movement. Please also add a segment about the sonidero movement and relationship with rock and roll."

response = client.models.generate_content(model=MODEL_ID, contents=prompt)
print(response.text)