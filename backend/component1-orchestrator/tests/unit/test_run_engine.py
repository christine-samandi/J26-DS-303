"""eval/run_engine.py: scores the golden set and catches a wrong expected value."""

import csv
import importlib.util
import json
import shutil
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parents[4] / "eval"

_spec = importlib.util.spec_from_file_location("run_engine", EVAL_DIR / "run_engine.py")
run_engine = importlib.util.module_from_spec(_spec)
sys.modules["run_engine"] = run_engine  # dataclasses look the module up here
_spec.loader.exec_module(run_engine)


def test_all_golden_scenarios_pass(tmp_path):
    assert run_engine.main(["--out", str(tmp_path)]) == 0

    (csv_file,) = tmp_path.glob("engine_*.csv")
    (md_file,) = tmp_path.glob("engine_*.md")
    rows = list(csv.DictReader(csv_file.open()))
    assert len(rows) == len(list((EVAL_DIR / "scenarios").glob("GS-*.json")))
    assert all(r["passed"] == "True" for r in rows)
    assert "Accuracy (verified scenarios)" in md_file.read_text()


def test_wrong_expected_value_is_reported(tmp_path):
    scenarios = tmp_path / "scenarios"
    scenarios.mkdir()
    src = EVAL_DIR / "scenarios" / "GS-001_band2_apit_underpaid.json"
    data = json.loads(src.read_text())
    data["expected"]["total_tax"] = 99999
    data["expected"]["steps"][3]["result"] = 1
    (scenarios / src.name).write_text(json.dumps(data))

    assert run_engine.main(["--scenarios", str(scenarios), "--out", str(tmp_path)]) == 1
    (row,) = csv.DictReader(next(tmp_path.glob("engine_*.csv")).open())
    assert row["passed"] == "False"
    assert "total_tax: expected 99999, got 96000" in row["mismatches"]
    assert "S3_BAND_2 result: expected 1, got 36000" in row["mismatches"]


def test_unverified_scenario_is_excluded_from_accuracy(tmp_path):
    scenarios = tmp_path / "scenarios"
    scenarios.mkdir()
    src = EVAL_DIR / "scenarios" / "GS-001_band2_apit_underpaid.json"
    shutil.copy(src, scenarios / src.name)
    data = json.loads(src.read_text())
    data["verification"]["status"] = "unverified"
    data["expected"]["total_tax"] = 1  # wrong, but unverified, so it must not fail the run
    (scenarios / "GS-999_draft.json").write_text(json.dumps({**data, "scenario_id": "GS-999"}))

    assert run_engine.main(["--scenarios", str(scenarios), "--out", str(tmp_path)]) == 0
    md = next(tmp_path.glob("engine_*.md")).read_text()
    assert "1/1 = 100.0%" in md
    assert "Unverified scenarios (excluded): 1" in md
