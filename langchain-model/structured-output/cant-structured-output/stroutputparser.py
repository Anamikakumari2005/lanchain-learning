from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-30B-A3B",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

template1 = PromptTemplate (template='write a 5 line detailed report on the following text .\n{topic}'
                           , input_variables=['topic'])
template2 = PromptTemplate (template='write 5 line a detailed report on the following text .\n{text}'
                           , input_variables=['text'])

prompt1 = template1.invoke({'topic':'black hole'})
result = model.invoke(prompt1)

prompt2 = template2.invoke({'text':result.content})
result1 = model.invoke(prompt2)

print(result1.content)