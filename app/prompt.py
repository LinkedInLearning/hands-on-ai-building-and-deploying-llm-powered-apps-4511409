##############################################################################
# Exercise 1:
# Please utilize Chainlit's app playground for prompt engineering and
# experimentation. Once done, modify the prompts template below with your
# newly developed prompts.
##############################################################################
from langchain.prompts import PromptTemplate


template = """Please act as an expert financial analyst when you answer the questions and pay special attention to the financial statements.  Operating margin is also known as op margin and is calculated by dividing operating income by revenue.
Given the following extracted parts of a long document and a question, create a final answer with references ("SOURCES").
If you don't know the answer, just say that you don't know. Don't try to make up an answer.
ALWAYS return a "SOURCES" field in your answer, with the format "SOURCES: <source1>, <source2>, <source3>, ...".

QUESTION: {question}
=========
{summaries}
=========
FINAL ANSWER:"""

PROMPT = PromptTemplate(
    template=template, input_variables=["summaries", "question"]
)

EXAMPLE_PROMPT = PromptTemplate(
    template="Content: {page_content}\nSource: {source}",
    input_variables=["page_content", "source"],
)
