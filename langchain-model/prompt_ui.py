from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typer import prompt
import streamlit as st
from langchain_core.prompts import PromptTemplate,load_prompt

load_dotenv()

model =  ChatGoogleGenerativeAI(model="gemini-3-flash-preview")
st.header("reasearch tool")

paper = st.selectbox(
    "Select Paper Type",
    ["Research Paper", "Essay", "Report"]
)

style = st.selectbox(
    "Select Writing Style",
    ["Formal", "Casual", "Technical"]
)

length = st.selectbox(
    "Select Length",
    ["Short", "Medium", "Long"]
)

template = load_prompt("template.json")
prompt = template.invoke({"paper": paper, "style": style, "length": length})

user_input = st.text_input("Enter your question here:")

if st.button("Submit"):
    result = model.invoke(prompt)
    st.write(result.text)