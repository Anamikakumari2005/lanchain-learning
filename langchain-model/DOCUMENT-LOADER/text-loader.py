from langchain_community.document_loaders import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

model =  ChatGoogleGenerativeAI(model="gemini-3-flash-preview")
parser = StrOutputParser()

load_dotenv()

prompt = PromptTemplate(
    template='write a summary for the following poen - \n {poem}',
    input_variables=['poem']
)

loader = TextLoader('cricker.txt')

docs = loader.load()

print(type(docs))
print(len(docs))
print(type(docs[0]))
print(docs[0].page_content)

chain = prompt | model | parser 

print(chain.invoke({'poem':docs[0].page_content}))