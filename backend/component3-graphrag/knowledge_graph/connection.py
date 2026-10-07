"""Neo4j connection helper.

Check your local database is reachable with:
    python -m knowledge_graph.connection
"""

from __future__ import annotations

from c3_common.config import get_settings


def get_driver():
    from neo4j import GraphDatabase  # imported lazily: only needed with the [graph] extra

    s = get_settings()
    if not s.neo4j_password:
        raise RuntimeError("NEO4J_PASSWORD is empty. Copy .env.example to .env and set it.")
    return GraphDatabase.driver(s.neo4j_uri, auth=(s.neo4j_user, s.neo4j_password))


def ping() -> str:
    with get_driver() as driver:
        driver.verify_connectivity()
        records, _, _ = driver.execute_query(
            "CALL dbms.components() YIELD name, versions RETURN name, versions[0] AS version"
        )
        return f"{records[0]['name']} {records[0]['version']}"


if __name__ == "__main__":
    print(f"Connected to {ping()} at {get_settings().neo4j_uri}")
