"""
orchestrator.py

Main pipeline coordinator.
Receives the list of uploaded PDFs and user questions, distributes
the work across the parallel runner, and collects structured results.
Built with LangGraph to manage agent state across steps.
"""
