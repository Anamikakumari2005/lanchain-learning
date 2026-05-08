
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
load_dotenv()
chat_history = [
    SystemMessage(content="You are a helpful assistant.")
]
model =  ChatGoogleGenerativeAI(model="gemini-3-flash-preview")

while True:
    user_input = input('you: ')
    chat_history.append(HumanMessage(content=user_input))
    if user_input == 'exit':
        break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content) )
    
    print("AI: ",result.content)
    
print("Chat history: ", chat_history)    