from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
chat_template = ChatPromptTemplate([
    ('system', 'you are helpful customer support assistant.'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human','{query}')
])

chat_history = []

with open('placeholder/chat_history.txt') as f:
    chat_history = []
    for line in f.readlines():
        chat_history.append(line.strip())

print(chat_history)    

prompt = chat_template.invoke({'chat_history': chat_history, 'query':HumanMessage(content='where my refund is? ') })
print(prompt)