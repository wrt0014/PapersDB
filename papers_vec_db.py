import os
import openai
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import warnings

warnings.filterwarnings("ignore")

openai.api_key = os.getenv("OPENAI_API_KEY")

loader = CSVLoader(file_path="nodes.csv", encoding='utf-8')

docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=0,
)

pages = splitter.split_documents(docs)

embedding = OpenAIEmbeddings()

persist_directory = 'vec_nodes'

vector_db = Chroma.from_documents(
    documents=pages,
    embedding=embedding,
    persist_directory=persist_directory
)
print("创建成功")