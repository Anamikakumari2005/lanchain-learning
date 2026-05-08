import langchain
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
load_dotenv()

llm = HuggingFaceEndpoint(      
    repo_id="Qwen/Qwen3-30B-A3B",
    task="text-generation"
)
parser = StrOutputParser()

model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template='write a joke {topic}',
    input_variables=['topic']
)
prompt2 = PromptTemplate(
    template='Explain the following joke - {text}',
    input_variables=['text']
)

chain = RunnableSequence(prompt,model,parser,prompt2,model,parser)

print(chain.invoke({'topic':'AI'}))
