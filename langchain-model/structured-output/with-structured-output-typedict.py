from typing import TypedDict , Annotated

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
load_dotenv()


model =  ChatGoogleGenerativeAI(model="gemini-3-flash-preview")

class Review(TypedDict):
    
    summary: Annotated[str, "A brief summary of the review"]
    sentiment: Annotated[str, "The sentiment of the review"]

structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""A girl named Riya lived in a small village. She was a very hardworking girl and always tried to learn something new. One day, she decided to improve her studies and practice English. In the beginning, she faced many problems. People laughed at her, and she felt shy. But she did not lose courage. After some months, she started speaking fluently, and everyone appreciated her. She proved that everything is possible with hard work and dedication.""")


print(result)