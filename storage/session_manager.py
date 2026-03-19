"""
session_manager.py

Session lifecycle manager.
Creates an isolated temporary directory for each user session,
tracks active sessions and their creation time, and exposes
methods to retrieve and invalidate session paths.
"""
