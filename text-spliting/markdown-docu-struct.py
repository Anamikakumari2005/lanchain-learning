from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

text = """
# 📄 PDF Chatbot using LangChain

## 🚀 Project Overview

This project is a **PDF-based Question Answering Chatbot** built using LangChain.
It allows users to upload a PDF and ask questions based on its content.

---

## 🧠 Features

* 📄 Load and read PDF files
* ✂️ Split large text into smaller chunks
* 🔍 Convert text into embeddings
* 📦 Store embeddings using FAISS (Vector Database)
* 🤖 Ask questions and get accurate answers from PDF

---

## 🛠️ Tech Stack

* Python 🐍
* LangChain
* OpenAI API
* FAISS
* PyPDF

---

## 📂 Project Structure

```
project/
│── main.py
│── example.pdf
│── README.md
```

---

## ⚙️ Installation

```bash
pip install langchain langchain-community langchain-openai faiss-cpu pypdf
```

---

## 🔑 Setup

Set your OpenAI API Key:

```python
import os
os.environ["OPENAI_API_KEY"] = "your_api_key_here"
```

---

## ▶️ Usage

Run the script:

```bash
python main.py
```

Then ask questions like:

```
What is the main topic of the PDF?
```

---

## 🔄 How It Works

1. Load PDF using PyPDFLoader
2. Split text using CharacterTextSplitter
3. Create embeddings using OpenAI
4. Store in FAISS vector database
5. Retrieve relevant chunks
6. Generate answer using LLM

---

## 💡 Example Output

```
Ask a question: What is AI?

Answer: Artificial Intelligence (AI) is the simulation of human intelligence...
```

---

## 📌 Future Improvements

* Add Streamlit UI
* Support multiple PDFs
* Add memory (chat history)
* Use open-source LLMs

---

## 🙌 Conclusion

This project demonstrates the power of **Retrieval-Augmented Generation (RAG)**
and is useful for building intelligent document-based assistants.

---

## 📎 Author

**Anamika Kumari**

"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN,
    chunk_size=400,
    chunk_overlap=0,
)

chunk = splitter.split_text(text)

print(len(chunk))
print(chunk[1])