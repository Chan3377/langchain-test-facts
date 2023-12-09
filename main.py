from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

# embeddings convert document into number
embeddings = OpenAIEmbeddings()

# split document into Chunk or small group of sentence with no more than {chunk_size} words
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=200,
    chunk_overlap=0,
)

# load and split document
loader = TextLoader("facts.txt")
docs = loader.load_and_split(
    text_splitter=text_splitter,
)

# setup database
# store embedding in chroma vector database
db = Chroma.from_documents(
    docs,
    embedding=embeddings,
    persist_directory="emb",
)

results = db.similarity_search(
    "What is an interesting fact about the English language?"
)

# print(docs)
for result in results:
    print("\n")
    print(result.page_content)
