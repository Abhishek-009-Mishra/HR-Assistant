import os

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Source HR documents to index
SOURCE_FILES = [
    "EmployeeHandbook.txt",
    "LeavePolicy.txt",
    "PythonDeveloperJD.txt",
]

docs = []
for filename in SOURCE_FILES:
    if not os.path.exists(filename):
        print(f"Warning: '{filename}' not found in current directory, skipping.")
        continue
    loader = TextLoader(filename, encoding="utf-8")
    docs.extend(loader.load())

if not docs:
    raise SystemExit(
        "No source documents were loaded. Make sure this script is run from "
        "the 'HR Assist' folder, next to the .txt files."
    )

# Split into smaller chunks for more precise retrieval
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = splitter.split_documents(docs)

vector_db = FAISS.from_documents(split_docs, embeddings)
vector_db.save_local("hr_vector_db")

print(
    f"hr_vector_db rebuilt successfully: "
    f"{len(split_docs)} chunks from {len(docs)} document(s)."
)
