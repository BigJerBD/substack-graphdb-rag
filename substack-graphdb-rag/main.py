import os

from langchain.chains import GraphCypherQAChain
from langchain_community.graphs import MemgraphGraph
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

graph = MemgraphGraph(
    url="bolt+ssc://3.139.66.247",
    username="bigjerbd@gmail.com",
    password="s7wvuzK77B4iuPK",
    database='memgraph'
)

cypher_prompt = """
Task:Generate Cypher statement to query a graph database.
Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
Schema:
{schema}
Note: Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
Do not include any text except the generated Cypher statement.
If the user asks about PS5, Play Station 5 or PS 5, that is the platform called PlayStation 5.

The question is:
{question}
"""

qa_prompt = """
You are an assistant that helps to form nice and human understandable answers.
The information part contains the provided information that you must use to construct an answer.
The provided information is authoritative, you must never doubt it or try to use your internal knowledge to correct it.
Make the answer sound as a response to the question. Do not mention that you based the result on the given information.
Here is an example:

Question: Which managers own Neo4j stocks?
Context:[manager:CTL LLC, manager:JANE STREET GROUP LLC]
Helpful Answer: CTL LLC, JANE STREET GROUP LLC owns Neo4j stocks.

Follow this example when generating answers.
If the provided information is empty, say that you don't know the answer.
Give an human-like answer.

Information:
{context}

Question: {question}
Helpful Answer
"""

chain = GraphCypherQAChain.from_llm(
    ChatOpenAI(
        temperature=0,
        openai_api_key='sk-substack-graph-rag-Rz185Gq5MsviijoV11pIT3BlbkFJJA1UgfxUOpEPGQOenUHl'
    ),
    graph=graph,
    cypher_prompt= PromptTemplate(
        input_variables=["schema", "question"], template=cypher_prompt
    ),
    qa_prompt= PromptTemplate(
        input_variables=["context", "question"], template=qa_prompt
    ),
    verbose=True,
    model_name="gpt-4o",
)

response = chain.invoke('How many term can be found inside the Protein Article')

print(f"Final response: {response['result']}")