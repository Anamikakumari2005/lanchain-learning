from langchain_core.prompts import PromptTemplate

template = PromptTemplate(
    template="""
You are an expert research assistant.

Write a detailed answer on the topic:

Paper Type: {paper}
Writing Style: {style}
Length: {length}

Instructions:
- Follow the selected paper type strictly
- Use the given writing style
- Maintain the required length
- Make the answer clear, structured, and informative
""",
    input_variables=["paper", "style", "length"],
    validate_template=True

)
template.save("template.json")