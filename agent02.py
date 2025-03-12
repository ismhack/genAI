import os
from dotenv import load_dotenv, find_dotenv

from langchain.prompts import PromptTemplate
from langchain_community.tools import TavilySearchResults
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import chain

from datetime import date

load_dotenv(find_dotenv())

gemini_key = os.getenv("GOOGLE_API_KEY")
tavily_key = os.getenv("TAVILY_API_KEY")
tavily_tool = TavilySearchResults(max_results=5)

MODEL_ID = "gemini-2.0-flash"
model = ChatGoogleGenerativeAI(model=MODEL_ID, temperature=0, google_api_key=gemini_key)

prompt = f"Metal bands news and events happening on {date.today()}. " \
         f"Including american and international bands. Provide source links to references information."

prompt = f"Rock and Roll bands like AC/DC Status Quo, Scorpions and Queen news and events happening on {date.today()}. " \
         f"Including american and international bands. Provide source links to references information."

#prompt = "El Tri rock and roll band from Mexico who recently celebrated 55 years of trajectory."

prompt = "The song break the rules by the Kiss The Status Quo why do people love it?" \
         " Fun facts about the lyrics and band "


# Planning: Create an outline for the essay
outline_template = PromptTemplate.from_template(
    "Create a detailed outline for a facebook post on {topic}"
)


# Research: Web search
def research_fn(topic):
    response = tavily_tool.invoke({"query": topic})
    return "\n".join([f"- {result['content']}" for result in response])


# Writing: Write the essay based on outline and research
writing_template = PromptTemplate.from_template(
    "Based on the following outline and research, write catchy and interesting post in spanish for facebook on '{"
    "topic}':\n\nOutline:\n{outline}\n\nResearch:\n{research}\n\nPost: "
)


@chain
def custom_chain(text):
    outline_prompt = outline_template.invoke({"topic": text})
    outline_output = model.invoke(outline_prompt)
    parsed_outline_output = StrOutputParser().invoke(outline_output)
    research_output = research_fn(text)
    output_write_prompt = writing_template.invoke(
        {"topic": text, "outline": parsed_outline_output, "research": research_output})
    return model.invoke(output_write_prompt)


essay = custom_chain.invoke(prompt)
print(essay.content)
