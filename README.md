# 🦜 LangChain Learning Project

A hands-on repository to explore and practice core **LangChain** concepts 
including chatbots, prompt templates, structured output, and text splitting.

## 📁 Project Structure

| File/Folder | Description |
|---|---|
| `chatbot.py` | Simple chatbot built with LangChain |
| `prompt_ui.py` | Prompt UI experiments |
| `prompt-template.py` | Prompt template practice |
| `structured-output/` | Structured output using Pydantic |
| `text-spliting/` | Text splitting techniques |

## 🛠️ Tech Stack

- Python
- LangChain
- OpenAI / Hugging Face
- LangSmith
- Streamlit

## ⚙️ Setup

1. Clone the repository
   git clone https://github.com/Anamikakumari2005/lanchain-learning.git

2. Install dependencies
   pip install -r requirements.txt

3. Create .env file
   cp .env.example .env
   # Add your API keys

4. Run
   python chatbot.py

## 🔐 Environment Variables

OPENAI_API_KEY=your_key_here
HUGGINGFACE_TOKEN=your_token_here
LANGSMITH_API_KEY=your_key_here
