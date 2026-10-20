# C3 demo: the knowledge graph keeps up with new law

Before the demo: Docker Desktop running (Neo4j), `.env` has `C3_MODE=graphrag`, the corpus is
downloaded, parsed and extracted (Phases 1-2). Open http://localhost:7474 in a browser tab.

## 1. The graph as it was known in early 2025

```bash
python -m knowledge_graph.build --until 2025-01-01
```
Builds the graph from the Acts certified before 2025 (2017 Act + amendments of 2021, 2022, 2023).

Neo4j Browser:
```cypher
MATCH p=(v:RuleVersion)-[:VERSION_OF]->(:Rule {rule_key:'personal_relief'})
OPTIONAL MATCH s=(v)-[:SUPERSEDES]->(:RuleVersion)
RETURN p, s
```
Three versions; V3 (Rs. 1,200,000) is `active`.

```bash
python -m retrieval_api.graph_backend "What personal relief can I claim?" --date 2025-06-01
```
Rs. 1,200,000: correct for what the system knows.

## 2. A new Act is published: Amendment Act No. 2 of 2025

```bash
python -m knowledge_graph.update DOC-IRA-AMEND-02-2025 --step
```
`--step` pauses after each stage (press Enter), so each one can be explained:
[1/5] parse the PDF, [2/5] extract entities, [3/5] rebuild the rule timelines,
[4/5] write only what is new to Neo4j, [5/5] change report, with the sentence from the Act that
caused each change:
```
+ document  Inland Revenue (Amendment) Act, No. 2 of 2025 ...: 20 chunks, 5 provisions amended
~ RULE-PERSONAL_RELIEF-V3    active -> replaced (valid to 2025-03-31)
+ RULE-PERSONAL_RELIEF-V4    Rs. 1,800,000  2025-04-01 .. now
~ RULE-INDIVIDUAL_SLABS-V3   active -> replaced (valid to 2025-03-31)
+ RULE-INDIVIDUAL_SLABS-V4   6%/18%/24%/30%/36% (5 bands)  2025-04-01 .. now
```

## 3. The graph updated itself; history is kept

Re-run the Neo4j query: V4 appears, with SUPERSEDES -> V3.

What the new Act changed in the 2017 Act (picture):
```cypher
MATCH (d:Document {document_id:'DOC-IRA-AMEND-02-2025'})<-[:PART_OF]-(:Provision)<-[:PART_OF]-(c:Chunk)
MATCH (c)-[a:AMENDS]->(p:Provision)
RETURN d, c, a, p
```

```bash
python -m retrieval_api.graph_backend "What personal relief can I claim?" --date 2025-06-01
python -m retrieval_api.graph_backend "What personal relief can I claim?" --date 2024-06-01
```
June 2025 -> Rs. 1,800,000 (new law); June 2024 -> still Rs. 1,200,000 (old law kept).

## 4. Not every Act changes every rule

```bash
python -m knowledge_graph.update DOC-IRA-AMEND-11-2026
```
"no change to the tracked rules": the 2026 Act changes other provisions, so no new versions.

## Other things to show

- Evidence: the API answer's `source.citation` and `evidence.matched_text` (http://localhost:8003/docs;
  after an update, `POST /reload` so a running API reads the new graph)
- Refusal: `python -m retrieval_api.graph_backend "customs duty on cars"` -> NO_MATCH
- Evaluation: `python -m evaluation.retrieval` (flat RAG vs GraphRAG)

After the demo the graph holds every Act again (the two updates added the rest).

## 5. Not only personal relief: every relief, and a document that triggers one

The same versioning covers every relief in Fifth Schedule paragraph 2 (general relief engine).

```bash
python -m knowledge_graph.versions        # 14/14 versions exact, including the six other reliefs
python -m retrieval_api.graph_backend "Can I claim relief for solar panels?" --date 2025-06-01
python -m retrieval_api.graph_backend "solar panel relief" --date 2020-06-01 --history
python -m retrieval_api.graph_backend "What reliefs can I claim?" --date 2021-06-01
```
- 2025: up to Rs. 600,000, limited to the panels' cost, with who can claim it and the citation
  (Act No. 10 of 2021, s.55(2), adding Fifth Schedule para 2(g)).
- 2020: no result, `NO_RULE_IN_FORCE_ON_DATE`; with `--history` it shows the relief starts on
  2021-04-01 (Table A of Act No. 10 of 2021).
- 2021: personal relief, expenditure relief (health, education, housing loan ...), rent relief
  and solar relief are all in force; in 2024 the expenditure relief is gone (ended 2022-12-31).

Neo4j Browser, all reliefs and their versions:
```cypher
MATCH (v:RuleVersion)-[:VERSION_OF]->(r:Rule {rule_type:'relief'})
RETURN r.title AS relief, v.rule_id, v.valid_from, v.valid_to, v.status, v.amount, v.cap,
       v.rate_percent ORDER BY relief, v.valid_from
```

