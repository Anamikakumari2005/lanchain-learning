import langchain
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableParallel , RunnableBranch , RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel , Field
from typing import Literal
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-30B-A3B",
    task="text-generation"
)
parser = StrOutputParser()

model1 = ChatHuggingFace(llm=llm)

class Feedback(BaseModel):
    
    sentiment : Literal['positive','negative'] = Field(description='give the sentiment of the feedback')

parser2 = PydanticOutputParser(pydantic_object=Feedback)
prompt1 = PromptTemplate(
    template= 'Classify the sentiment of the following feedback text into positive or negative \n {feedback} \n {format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction':parser2.get_format_instructions()}
)


classifire_chain = prompt1 | model1 | parser2

prompt2 = PromptTemplate(
    template='write an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template='write an appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)

branch_chain = RunnableBranch(
    (lambda x:x.sentiment == 'positive', prompt2 | model1 | parser),
    (lambda x:x.sentiment == 'negative', prompt3 | model1 | parser),
    RunnableLambda(lambda x:"could not find sentiment")
)

chain = classifire_chain | branch_chain

print(chain.invoke({'feedback': 'this is a beautiful i am happy  phone'}))

chain.get_graph().print_ascii()