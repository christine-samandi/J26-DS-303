"""Smoke test: the C3 packages import and settings resolve."""

import importlib

import pytest

from c3_common.config import get_settings


@pytest.mark.parametrize(
    "module",
    [
        "data_collection",
        "entity_recognition",
        "knowledge_graph",
        "vector_store",
        "retrieval_api",
        "j26_contracts.retrieval",
    ],
)
def test_packages_import(module):
    importlib.import_module(module)


def test_contracts_folder_found():
    schemas = get_settings().contracts_dir / "schemas"
    assert (schemas / "retrieval_request.schema.json").is_file()
    assert (schemas / "retrieval_output.schema.json").is_file()
