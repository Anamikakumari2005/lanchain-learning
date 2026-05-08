from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import chain
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.output_parsers import  JsonOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-30B-A3B",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()

template = PromptTemplate(
    template= ' give me the name , age and city of a fictional person \n {format_instruction}',
    input_variables=[],
    partial_variables={'format_instruction' : parser.get_format_instructions() }
)

prompt = template.format()
result = model.invoke(prompt)
final = parser.parse(result.content)
print(final.content)