"""
query.py

Pydantic models for query-related data structures.

Models:
- UserQuestion: a question defined by the user with an optional label.
- Citation: the source fragment the agent used to build an answer.
- AgentAnswer: the full agent response for one question on one document,
  including the answer text, citation and confidence score.
"""
