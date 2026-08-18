#!/bin/bash
set -e

echo "Creating folder structure for J26-DS-303 Tax Assistant..."

# --- Top level ---
mkdir -p contracts/schemas
mkdir -p contracts/mock_data/ocr_output_examples
mkdir -p contracts/mock_data/retrieval_output_examples
mkdir -p contracts/mock_data/pipeline_trace_examples
mkdir -p contracts/mock_data/assurance_verdict_examples

# --- Frontend ---
mkdir -p frontend/src/components
mkdir -p frontend/src/pages
mkdir -p frontend/src/services
mkdir -p frontend/src/types
mkdir -p frontend/public
mkdir -p frontend/tests

# --- Backend: Component 1 (Orchestrator) ---
mkdir -p backend/component1-orchestrator/orchestrator
mkdir -p backend/component1-orchestrator/tax_calculation_agent
mkdir -p backend/component1-orchestrator/explanation_agent
mkdir -p backend/component1-orchestrator/api
mkdir -p backend/component1-orchestrator/tests/unit
mkdir -p backend/component1-orchestrator/tests/integration

# --- Backend: Component 2 (OCR) ---
mkdir -p backend/component2-ocr/preprocessing
mkdir -p backend/component2-ocr/ocr_engines
mkdir -p backend/component2-ocr/document_classifier
mkdir -p backend/component2-ocr/entity_extraction
mkdir -p backend/component2-ocr/api
mkdir -p backend/component2-ocr/tests
mkdir -p backend/component2-ocr/sample_documents

# --- Backend: Component 3 (GraphRAG) ---
mkdir -p backend/component3-graphrag/data_collection
mkdir -p backend/component3-graphrag/entity_recognition
mkdir -p backend/component3-graphrag/knowledge_graph
mkdir -p backend/component3-graphrag/vector_store
mkdir -p backend/component3-graphrag/retrieval_api
mkdir -p backend/component3-graphrag/tests
mkdir -p backend/component3-graphrag/raw_regulatory_docs

# --- Backend: Component 4 (Assurance) ---
mkdir -p backend/component4-assurance/consistency_checker
mkdir -p backend/component4-assurance/llm_judge
mkdir -p backend/component4-assurance/citation_verifier
mkdir -p backend/component4-assurance/drift_monitor
mkdir -p backend/component4-assurance/decision_engine
mkdir -p backend/component4-assurance/api
mkdir -p backend/component4-assurance/tests

# --- Docs ---
mkdir -p docs/conceptual_diagram
mkdir -p docs/panel_defense_notes

echo "Folders created. Adding .gitkeep + placeholder READMEs..."

# .gitkeep for genuinely empty leaf folders (git won't track empty dirs)
find . -type d -empty -not -path "./.git*" -exec touch {}/.gitkeep \;

# Component READMEs
cat > frontend/README.md << 'EOF'
# Frontend
Owner: Fernando (Component 1)

## How to run standalone
(Add setup instructions once scaffolded — React/Next.js or Streamlit)

Talks only to `backend/component1-orchestrator/api`.
EOF

cat > backend/component1-orchestrator/README.md << 'EOF'
# Component 1 — Agentic Orchestrator, Tax Calculation Agent, Explanation Agent
Owner: Fernando (IT23180420)

## How to run standalone
Test against contracts/mock_data/ for Comp2 and Comp3 outputs before real integration.
EOF

cat > backend/component2-ocr/README.md << 'EOF'
# Component 2 — OCR & Document Understanding
Owner: Senevirathne (IT23271296)

## How to run standalone
Input: image in `sample_documents/`
Output: JSON matching `contracts/schemas/ocr_output.schema.json`
EOF

cat > backend/component3-graphrag/README.md << 'EOF'
# Component 3 — GraphRAG Knowledge Base & Regulatory Retrieval
Owner: Elvitigala (IT23280724)

## How to run standalone
Output: JSON matching `contracts/schemas/retrieval_output.schema.json`
EOF

cat > backend/component4-assurance/README.md << 'EOF'
# Component 4 — Output Assurance & Drift Monitoring
Owner: De Zoysa (IT22268976)

## How to run standalone
Input: pipeline trace matching `contracts/schemas/pipeline_trace.schema.json`
Output: verdict matching `contracts/schemas/assurance_verdict.schema.json`
EOF

cat > contracts/CONTRACT_CHANGELOG.md << 'EOF'
# Contract Changelog
Log every change to any schema in contracts/schemas/ here, with date, author, and reason.
Any schema change must be reviewed by all components that depend on it.
EOF

echo "Done. Run 'git status' to review, then add/commit/push."
