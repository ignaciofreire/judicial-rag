"""
parallel_runner.py

Parallel execution manager for PDF processing.
Uses asyncio with a ThreadPoolExecutor to process multiple PDFs
concurrently. A semaphore limits the number of simultaneous
extractions to avoid memory exhaustion.
"""
