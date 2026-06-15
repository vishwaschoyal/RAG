"""
=========================================================
RAG Pipeline using LangChain + FAISS
=========================================================

Steps:
1. Load the document
2. Split it into chunks
3. Generate embeddings
4. Create a FAISS vector store
5. Save the vector database
6. Retrieve relevant chunks using MMR
7. Perform similarity search with scores

Author: Vishwas Choyal
"""

import os
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# =========================================================
# Load Environment Variables
# =========================================================

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# =========================================================
# Configuration
# =========================================================

DOCUMENT_PATH = "text.txt"
VECTOR_DB_PATH = "local_vec_db"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

QUERY = "Who is Vishwas Choyal?"

# =========================================================
# Load Document
# =========================================================

loader = TextLoader(DOCUMENT_PATH)
documents = loader.load()

# =========================================================
# Split Documents into Chunks
# =========================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)

chunked_documents = text_splitter.transform_documents(documents)

print(f"Total chunks created: {len(chunked_documents)}")

# =========================================================
# Create Embeddings and Vector Store
# =========================================================

embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-large"
)

vector_db = FAISS.from_documents(
    chunked_documents,
    embedding_model,
)

# Save locally so embeddings don't need to be regenerated
vector_db.save_local(VECTOR_DB_PATH)

# =========================================================
# Create Retriever (MMR)
# =========================================================

retriever = vector_db.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 8,
        "fetch_k": 20,
    },
)

retrieved_docs = retriever.invoke(QUERY)

print("\n================ Retrieved Chunks ================\n")

for i, doc in enumerate(retrieved_docs, start=1):
    print(f"Chunk {i}")
    print("-" * 60)
    print(doc.page_content)
    print()

# =========================================================
# View Stored Chunks
# =========================================================

print("\n================ Stored Chunks ================\n")

for i, doc in enumerate(chunked_documents[:10], start=1):
    print(f"Chunk {i}")
    print("-" * 60)
    print(doc.page_content)
    print()

# =========================================================
# Similarity Search with Scores
# =========================================================

results = vector_db.similarity_search_with_score(QUERY)

print("\n=========== Similarity Search Scores ===========\n")

for i, (doc, score) in enumerate(results, start=1):
    print(f"Result {i}")
    print(f"Distance Score: {score}")
    print(doc.page_content)
    print("-" * 60)

# =========================================================
# Notes
# =========================================================

"""
MMR (Maximal Marginal Relevance)
--------------------------------

Normal similarity search:
Query
    ↓
Returns the most similar chunks

Problem:
Many retrieved chunks may contain nearly identical information.

MMR:
Query
    ↓
Fetch fetch_k candidate chunks
    ↓
Select k chunks that are:
    • Relevant to the query
    • Diverse from each other

Example:

Without MMR:
- Education
- Education
- Education

With MMR:
- Education
- Skills
- Projects

This provides richer context for RAG systems.
"""

# In FAISS:
# similarity_search_with_score() returns DISTANCE values.
# Lower distance ⇒ Higher similarity.