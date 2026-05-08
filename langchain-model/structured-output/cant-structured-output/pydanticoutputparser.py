from typing import TypedDict , Annotated
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from typing import Optional
load_dotenv()


model =  ChatGoogleGenerativeAI(model="gemini-3-flash-preview")

class person (BaseModel):
    name:str = Field(description='Name of the person')
    age: int = Field(gt=18, description='Age of the person (must be greater than 18)')
    city : str = Field(description='Name of the city where person are belongs')
    
parser = PydanticOutputParser(pydantic_object=person)

template = PromptTemplate(
    template='generate the name , age and city of a fictional {place} person \n{format_instruction}',
    input_variables=['place'],
    partial_variables={
        'format_instruction': parser.get_format_instructions()
    }
)
    
    
chain = template | model | parser    
finalb = chain.invoke({'place':'india'})

print(finalb)

    
# prompt1 = template.invoke({'place':'india'})
# result = model.invoke(prompt1)
# try:
#     if isinstance(result.content, list):
#         text_output = result.content[0]["text"]
#     else:
#         text_output = result.content

#     final = parser.parse(text_output)
#     print(final)

# except Exception as e:
#     print("Parsing Error:", e)
#     print("Raw Output:", result.content)