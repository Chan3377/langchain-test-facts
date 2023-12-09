from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from redundant_filter_retriever import RedundantFilterRetriever
from dotenv import load_dotenv

load_dotenv()

# llm model
chat = ChatOpenAI()
# use for converting doc to chunk of text
embeddings = OpenAIEmbeddings()
# setup database
db = Chroma(
    persist_directory="emb",
    embedding_function=embeddings,
)

# use to remove duplicate text/value
retriever = RedundantFilterRetriever(
    embeddings=embeddings,
    chroma=db,
)

# A retriever is an object that can take in a string and return some relevant documents
# retriever = db.as_retriever()

# chain the function together
# chain_type="stuff" - take some context from the vector store and "stuff" it into the prompt
chain = RetrievalQA.from_chain_type(
    llm=chat,
    retriever=retriever,
    chain_type="stuff",
)

result = chain.run("What is an interesting fact about the English language?")

print(result)
