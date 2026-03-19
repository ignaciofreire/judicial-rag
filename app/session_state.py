"""
session_state.py

Centralises all Streamlit session state initialisation and access.
Keeps session variables (uploaded files, questions, results, session_id)
in one place to avoid scattered st.session_state references across components.
"""
