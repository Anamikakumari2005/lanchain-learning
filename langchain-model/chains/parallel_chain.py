import langchain
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableParallel

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-30B-A3B",
    task="text-generation"
)

model1 = ChatHuggingFace(llm=llm)
model2 =  ChatGoogleGenerativeAI(model="gemini-3-flash-preview")

prompt1 = PromptTemplate(
    template='generate a short and simple notes from the following text \n {text}',
    input_variables=['text']
)
prompt2 = PromptTemplate(
    template='generate a 5 point short question  from the following text on \n {text}',
    input_variables=['text']
)
prompt3 = PromptTemplate(
    template='merge the provided notes and quiz into single document \n notes -> {notes} ans quiz -> {quiz}',
    input_variables=['notes','quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes': prompt1 | model1 | parser,
    'quiz' : prompt2 | model2 | parser
})

merge_chain = prompt3 | model1 | parser

final_chain = parallel_chain | merge_chain

text = """
Python is a high-level, general-purpose programming language that emphasizes code readability, simplicity, and ease-of-writing with the use of significant indentation,[38] "plain English" naming, an extensive ("batteries-included") standard library, and garbage collection. Python supports multiple programming paradigms but with an emphasis on object-oriented programming and dynamic typing.

Guido van Rossum began working on Python in the late 1980s as a successor to the ABC programming language. Python 3.0, released in 2008, was a major revision and not completely backward-compatible with earlier versions. Beginning with Python 3.5,[39] capabilities and keywords for typing were added to the language, allowing optional static typing.[40] As of 2026, the Python Software Foundation supports Python 3.10, 3.11, 3.12, 3.13, and 3.14, following the project's annual release cycle and five-year support policy. Python 3.15 is currently in the alpha development phase, and the stable release is expected to come out in October 2026.[41] Earlier versions in the 3.x series have reached end-of-life and no longer receive security updates.

Python has gained widespread use in the machine learning community.[42][43][44][45] It is widely taught as an introductory programming language.[46] Since 2003, Python has consistently ranked in the top ten of the most popular programming languages in the TIOBE Programming Community Index, which ranks based on searches in 24 platforms.[47]
"""

result = final_chain.invoke({'text':text})

print(result)