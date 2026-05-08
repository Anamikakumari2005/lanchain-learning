import langchain
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence , RunnableParallel
load_dotenv()

llm = HuggingFaceEndpoint(      
    repo_id="Qwen/Qwen3-30B-A3B",
    task="text-generation"
)
parser = StrOutputParser()

model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template='generate a tweet about {topic}',
    input_variables=['topic']
)

prompt1 = PromptTemplate(
    template='generate a Linkedin post about {topic}',
    input_variables=['topic']
)

parallel_chain = RunnableParallel({
    'tweet' : RunnableSequence(prompt,model,parser),
    'linkedin': RunnableSequence(prompt1,model,parser)
})

print(parallel_chain.invoke({'topic':'AI'}))