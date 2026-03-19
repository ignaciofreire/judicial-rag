"""
cleanup.py

Cleanup routines for temporary session data.
Deletes the session temporary directory and its ChromaDB collection
when a session ends or when the configured timeout is exceeded.
Designed to run both on explicit logout and on a background schedule.
"""
