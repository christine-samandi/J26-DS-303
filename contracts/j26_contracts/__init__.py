"""Shared data contracts for the J26-DS-303 tax assistant.

The Pydantic models in this package ARE the contracts between components. The JSON Schema
files in contracts/schemas/ are generated from them (python -m j26_contracts.export).

Install once per component virtual environment:
    pip install -e <path-to-repo>/contracts

Use:
    from j26_contracts.retrieval import RetrievalRequest, RetrievalOutput
"""
