from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """
Agar tum PDF ya document read kar rahi ho (jaise tumne PyPDFLoader use kiya tha), to:

Pehle document load hoga
Fir uska text split hoga
Fir embeddings / LLM ko diya jayega Short Summary
Ye import line ek text splitting tool ko use karne ke liye hai
Use: large text → small chunks
"""
splitter = RecursiveCharacterTextSplitter(
    chunk_size=30,
    chunk_overlap=0
)

chunk = splitter.split_text(text)

print(len(chunk))
print(chunk)