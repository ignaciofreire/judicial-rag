"""
rag_agent.py

RAG agent responsible for answering user questions.
For each question, retrieves the most relevant chunks from the vector
store, sends them to the LLM with the question, and returns the answer
along with the source fragment and similarity score.
"""
