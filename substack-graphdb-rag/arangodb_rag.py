import os

from arango import ArangoClient

from langchain_community.graphs import ArangoGraph
from langchain.chains import ArangoGraphQAChain
from langchain_openai import ChatOpenAI


question='...'

db = ArangoClient(hosts=os.environ['ARANGO_ENDPOINT']).db(
    os.environ["ARANGO_DATABASE"],
    os.environ["ARANGO_USERNAME"],
    os.environ["ARANGO_PASSWORD"],
    verify=True
)

graph = ArangoGraph(db)
graph.set_schema()

chain = ArangoGraphQAChain.from_llm(
    ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        openai_api_key=os.environ["OPENAI_API_KEY"]
    ),
    graph=graph,
    verbose=True
)

print(chain.run(question))
