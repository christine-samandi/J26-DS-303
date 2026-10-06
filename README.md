# Explainable Agentic AI Tax Assistant for Sri Lanka

**Project ID:** J26-DS-303 
**Module:** IT4010 – Research Project 

An explainable, multi-agent AI assistant that helps individuals in Sri Lanka understand and comply with personal income tax under the **Inland Revenue Act No. 24 of 2017**. Every figure it produces is calculated deterministically and every explanation is linked to the exact legal provision it relies on.

---

## ✨ Key Ideas

- **Hybrid reasoning:** a deterministic rule engine does all the arithmetic. The LLM handles only language and interpretation, so it never computes the tax.
- **Citation-linked explainability:** each step of the calculation is traced back to the relevant section of the Act.
- **Multi-agent architecture:** an orchestrator coordinates specialized agents for documents, retrieval, calculation, and assurance.
- **Confidence-scored escalation:** low-confidence results are flagged instead of being guessed.
- **Privacy by design:** PII is redacted before LLM calls and restored afterwards, using spaCy NER.

---

## 🧩 Components & Team

| # | Component | Owner | Student ID |
|---|-----------|-------|------------|
| 1 | Agentic Orchestrator, Tax Calculation Agent, XAI Layer & Frontend | Fernando W I C S | IT23180420 |
| 2 | OCR & Document Understanding | Senevirathne R S N N| IT23271296 |
| 3 | GraphRAG Knowledge Base & Regulatory Retrieval | Elvitigala C S | IT23280724 |
| 4 | Output Assurance & Drift Monitoring | De Zoysa K R D P | IT22268976 |

---

## 🏗️ Architecture

```
            ┌──────────────────────────┐
            │   Frontend (User / UI)   │
            └────────────┬─────────────┘
                         │
            ┌────────────▼─────────────┐
            │   Agentic Orchestrator   │  ← C1
            └──┬──────┬──────┬──────┬──┘
               │      │      │      │
        ┌──────▼─┐ ┌──▼────┐ ┌▼─────┐ ┌▼──────────┐
        │  OCR   │ │GraphRAG│ │ Tax  │ │  Output   │
        │  (C2)  │ │  (C3)  │ │ Calc │ │ Assurance │
        │        │ │        │ │ (C1) │ │   (C4)    │
        └────────┘ └────────┘ └──────┘ └───────────┘
                         │
              XAI / Explanation Layer (C1)
```

All components communicate through **frozen JSON contracts** in `contracts/`, so the team can develop in parallel against mock data.

---

## 📁 Repository Structure

```
.
├── contracts/            # Shared JSON schemas (frozen; changes need full-team approval)
│   ├── ocr_output.json
│   ├── retrieval_output.json
│   ├── pipeline_trace.json
│   └── assurance_verdict.json
├── orchestrator/         # C1 – agent orchestration, tax engine, XAI layer
├── frontend/             # C1 – user interface
├── ocr/                  # C2 – document understanding
├── graphrag/             # C3 – knowledge graph & retrieval
├── assurance/            # C4 – output assurance & drift monitoring
├── mocks/                # Mock data for independent development
├── docs/                 # Reports, diagrams, references
└── README.md
```

---

## 🌿 Branching Strategy

We use **short-lived feature branches, prefixed by component**.

- `main` is protected and always stable. Nobody pushes to it directly; all work reaches it through Pull Requests.
- Feature branches follow the pattern `<component>/<type>/<short-description>`:

```
orchestrator/feat/tax-calc-engine
ocr/feat/payslip-parser
graphrag/fix/citation-lookup
assurance/feat/drift-monitor
contracts/update/retrieval-output-v2
```

**Types:** `feat`, `fix`, `refactor`, `docs`, `test`, `chore`

### Rules
1. Branch off `main`, keep branches small, and merge within a few days.
2. All merges go through a **Pull Request** with at least **1 reviewer**.
3. Changes to `contracts/` require approval from **all four members** (enforced via `CODEOWNERS`).
4. Sync with `main` regularly to avoid painful conflicts.
5. Milestones are tagged on `main` (e.g. `v0.5-pp1`).

### Commit messages
```
<type>(<component>): short description

feat(ocr): add payslip field extraction
fix(orchestrator): correct APIT relief calculation
```

---

## 🚀 Getting Started

```bash
# Clone the repo
git clone <repo-url>
cd <repo-name>

# Set up your component (example)
cd orchestrator
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> Each component folder has its own `README.md` with specific setup and run instructions.

---

## 🗓️ Milestones

| Milestone | Date | Target |
|-----------|------|--------|
| Proposal Presentation | Sep 2026 | ✅ Done |
| Proposal Report | Sep 2026 | ✅ Done |
| Progress Presentation 1 | 22 Oct 2026 | 50% implementation |
| Progress Checklist | End Oct 2026 | — |

---

## 📚 Knowledge Source

- Inland Revenue Act No. 24 of 2017 (and amendments), [Inland Revenue Department, Sri Lanka](https://www.ird.gov.lk)

