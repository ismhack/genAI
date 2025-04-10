import os
from dotenv import load_dotenv, find_dotenv

from langchain.prompts import PromptTemplate
from langchain_community.tools import TavilySearchResults
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import chain

from datetime import date

from spotify.browse import top_10_songs_by_artist_name, top_10_songs_by_playlist

load_dotenv(find_dotenv())

gemini_key = os.getenv("GOOGLE_API_KEY")
tavily_key = os.getenv("TAVILY_API_KEY")
tavily_tool = TavilySearchResults(max_results=5)

MODEL_ID = "gemini-2.0-flash"
model = ChatGoogleGenerativeAI(model=MODEL_ID, temperature=0, google_api_key=gemini_key)

prompt = "80s trash metal"
# Planning: Create an outline for the essay
outline_template = PromptTemplate.from_template(
    "Create a detailed outline for a facebook post about a the {topic} playlist"
)


# Research: Web search
def research_fn(topic):
    response = tavily_tool.invoke({"query": topic})
    return "\n".join([f"- {result['content']}" for result in response])


# top 10 songs by artist
def top_songs_fn(artist):
    response = top_10_songs_by_playlist(artist)
    return max(response, key=lambda key: response[key])


# Writing: Write the essay based on outline and research
writing_template = PromptTemplate.from_template(
    "Based on the following outline a playlist "
    "and the research about the playlist {topic} "
    "write catchy and interesting post in spanish for facebook on '{topic} playlist and its top song {song}':\n\nOutline:\n{"
    "outline}\n\nResearch:\n{research}\n\nPost: "
)


@chain
def custom_chain(text):
    outline_prompt = outline_template.invoke({"topic": text})
    outline_output = model.invoke(outline_prompt)
    parsed_outline_output = StrOutputParser().invoke(outline_output)
    top_song = top_songs_fn(text)
    research_output = research_fn(f"{text}")
    output_write_prompt = writing_template.invoke(
        {"topic": text, "song": {top_song}, "outline": parsed_outline_output, "research": research_output})
    return model.invoke(output_write_prompt)


essay = custom_chain.invoke(prompt)
print(essay.content)
