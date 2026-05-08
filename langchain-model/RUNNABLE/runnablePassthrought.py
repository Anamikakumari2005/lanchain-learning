import langchain
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnablePassthrough
load_dotenv()

model =  ChatGoogleGenerativeAI(model="gemini-3-flash-preview")
parser = StrOutputParser()


prompt = PromptTemplate(
    template='write a joke {topic}',
    input_variables=['topic']
)
prompt2 = PromptTemplate(
    template='Explain the following joke - {text}',
    input_variables=['text']
)

joke_gen_chain = RunnableSequence(prompt,model,parser)

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'explain':RunnableSequence(prompt2,model,parser)
})

chain = RunnableSequence(joke_gen_chain,parallel_chain)

print(chain.invoke({'topic':'AI'}))
