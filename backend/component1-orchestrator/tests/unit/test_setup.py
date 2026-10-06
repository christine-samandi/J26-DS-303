"""Smoke test: runtime dependencies from pyproject.toml are installed."""

import pydantic
import yaml


def test_runtime_dependencies_installed():
    assert pydantic.VERSION.startswith("2.")
    assert yaml.safe_load("a: 1") == {"a": 1}
