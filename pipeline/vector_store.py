"""
vector_store.py

ChromaDB interface for storing and querying embeddings.
Creates an isolated collection per session_id so that documents
from different users never mix. Exposes methods for insertion,
similarity search and collection cleanup.
"""
