"""eval/run_explanation_checks.py: tampers really tamper, harmless rewrites stay correct,
and the scoring runs through the real post-check, reproducibly."""

import csv
import sys
from pathlib import Path

import pytest

EVAL_DIR = Path(__file__).resolve().parents[4] / "eval"
sys.path.insert(0, str(EVAL_DIR))
import run_explanation_checks as rec  # noqa: E402

from explanation_agent.post_check import TAG, numbers_in  # noqa: E402

GS001 = EVAL_DIR / "scenarios" / "GS-001_band2_apit_underpaid.json"
TAMPERS = [g for g in rec.GENERATORS if not g[0]]
HARMLESS = [g for g in rec.GENERATORS if g[0]]


@pytest.fixture(scope="module")
def gs001():
    return rec.explain_scenario(rec.load_scenario(GS001))


@pytest.fixture(scope="module")
def explained():
    return rec.load_explained(EVAL_DIR / "scenarios")


def _figures(lines):
    return sorted(n for _, t in lines for n in numbers_in(t))


def _tags(lines):
    return sorted(tag for _, t in lines for tag in TAG.findall(t))


def test_every_untampered_template_passes(explained):
    assert len(explained) == len(list((EVAL_DIR / "scenarios").glob("GS-*.json")))
    assert all(rec.run_check(e, e.lines).passed for e in explained)


def test_a_failing_template_stops_the_run(tmp_path, monkeypatch):
    (tmp_path / GS001.name).write_bytes(GS001.read_bytes())
    monkeypatch.setattr(rec, "run_check", lambda e, lines: rec.check("not json", (), (), ()))
    with pytest.raises(SystemExit, match="untampered template fails"):
        rec.load_explained(tmp_path)


@pytest.mark.parametrize(("harmless", "type_", "subtype", "generate"), rec.GENERATORS)
def test_no_candidate_is_the_untampered_text(explained, harmless, type_, subtype, generate):
    for e in explained:
        assert all(c.lines != e.lines for c in generate(e))


def test_small_amount_changes_only_the_result(gs001):
    (c, *_) = [c for c in rec.amount_small(gs001) if c.step_id == "S6_PAYABLE"]
    assert (
        dict(c.lines)["S6_PAYABLE"]
        == "Tax still payable: total tax less APIT credit: Rs. 46,001.00"
    )
    assert [x for x in c.lines if x[0] != "S6_PAYABLE"] == [
        x for x in gs001.lines if x[0] != "S6_PAYABLE"
    ]


def test_digit_swap_never_makes_a_leading_zero(explained):
    for e in explained:
        for c in rec.amount_digit_swap(e):
            new = c.detail.split(" -> ")[1]
            assert not new.startswith("0") or new.startswith("0.")


def test_role_swap_keeps_every_figure_allowed_but_moves_the_result(gs001):
    c = next(
        c for c in rec.amount_role_swap(gs001) if c.detail.startswith("result 46,000.00 <-> total")
    )
    assert dict(c.lines)["S6_PAYABLE"] == (
        "Tax still payable: total tax less APIT credit (total tax: Rs. 46,000.00): Rs. 96,000.00"
    )


def test_role_swap_labels_have_no_digits_or_number_words():
    assert rec._label("band_1_tax") == "band tax"
    assert rec._label("rate_percent") == "rate"


@pytest.mark.parametrize(
    ("generator", "expected"),
    [
        (rec.rounding_cents, "Rs. 96,000.50"),
        (rec.rounding_whole, "Rs. 95,999"),
        (rec.rounding_one_figure, "Rs. 100,000"),
    ],
)
def test_rounding_tweaks_on_total_tax(gs001, generator, expected):
    (c,) = [c for c in generator(gs001) if c.step_id == "S4_TOTAL_TAX"]
    assert expected in dict(c.lines)["S4_TOTAL_TAX"]


def test_citation_tampers_change_tags_and_not_figures(gs001):
    for generate in (
        rec.citation_swap,
        rec.citation_wrong_rule,
        rec.citation_invented,
        rec.drop_citation,
        rec.wrong_year_cited_rule,
    ):
        for c in generate(gs001):
            assert _figures(c.lines) == _figures(gs001.lines)
            assert _tags(c.lines) != _tags(gs001.lines) or c.lines != gs001.lines


def test_wrong_year_cited_rule_names_the_other_year(gs001):
    assert all("YA2024_25" in c.detail.split(" -> ")[1] for c in rec.wrong_year_cited_rule(gs001))


def test_drop_step_removes_exactly_one_line(gs001):
    for c in rec.drop_step(gs001):
        assert len(c.lines) == len(gs001.lines) - 1
        assert c.step_id not in dict(c.lines)


@pytest.mark.parametrize("generate", [g[3] for g in HARMLESS])
def test_harmless_rewrites_keep_every_figure_and_citation(explained, generate):
    for e in explained:
        for c in generate(e):
            assert [sid for sid, _ in c.lines] == [sid for sid, _ in e.lines]
            assert _tags(c.lines) == _tags(e.lines)
            for (_, new), step in zip(c.lines, e.steps, strict=True):
                assert step.result in numbers_in(new)


def test_correct_year_mention_names_the_scenarios_own_year(gs001):
    assert all(
        "for YA 2025/26:" in t
        for c in rec.correct_year_mention(gs001)
        for sid, t in c.lines
        if sid == c.step_id
    )


def test_paraphrases_pass_the_real_post_check(explained):
    for e in explained:
        for _, _, _, generate in [g for g in HARMLESS if g[1] == "paraphrase"]:
            for c in generate(e):
                assert rec.run_check(e, c.lines).passed, (e.scenario_id, c.detail)


def test_a_changed_payable_is_caught(gs001):
    (c,) = [
        c
        for c in rec.amount_large(gs001)
        if c.step_id == "S6_PAYABLE" and c.detail == "46,000.00 -> 56,000.00"
    ]
    v = rec.Variant("GS-001", False, "amount", "large", 0, c)
    s = rec.score(gs001, v)
    assert s.outcome == "caught" and s.reason == "hallucination"
    assert any("56000.00 is not one of the engine's figures" in f for f in s.failures)


def test_a_dropped_step_is_a_structural_catch(gs001):
    v = rec.Variant("GS-001", False, "drop", "step", 0, rec.drop_step(gs001)[0])
    assert rec.score(gs001, v).reason == "structural"


def test_not_applicable_is_not_scored(gs001):
    s = rec.score(gs001, rec.Variant("GS-001", False, "amount", "digit_swap", 0, None))
    assert s.outcome == "not_applicable"
    assert rec.rate("x", [s]).total == 0


def test_variants_are_sampled_without_repeats(explained):
    for e in explained:
        for v_type in {(v.type, v.subtype) for v in rec.variants(e, 0, 3)}:
            picked = [v.candidate for v in rec.variants(e, 0, 3) if (v.type, v.subtype) == v_type]
            assert len(picked) <= 3
            assert len({c.lines for c in picked if c}) == len([c for c in picked if c])


def test_same_seed_same_variants_and_scores(explained):
    one = rec.score_all(explained, seed=0, per_type=3)
    two = rec.score_all(explained, seed=0, per_type=3)
    assert one == two
    other = rec.score_all(explained, seed=1, per_type=3)
    assert [s.variant.candidate for s in other] != [s.variant.candidate for s in one]


def test_main_writes_csv_and_markdown(tmp_path):
    assert rec.main(["--out", str(tmp_path)]) == 0

    (csv_file,) = tmp_path.glob("explanation_checks_*.csv")
    (md_file,) = tmp_path.glob("explanation_checks_*.md")
    rows = list(csv.DictReader(csv_file.open(encoding="utf-8")))
    assert {r["outcome"] for r in rows} <= {
        "caught",
        "missed",
        "passed",
        "false_alarm",
        "not_applicable",
    }
    assert {(r["type"], r["subtype"]) for r in rows} == {(g[1], g[2]) for g in rec.GENERATORS}
    md = md_file.read_text(encoding="utf-8")
    for heading in (
        "Overall catch rate (all tamper types):",
        "False-alarm rate: paraphrase:",
        "False-alarm rate: correct_year_mention:",
        "Git SHA:",
        "## Misses",
        "## False alarms",
    ):
        assert heading in md


def test_rates_count_catches_and_false_alarms():
    def s(harmless, flagged):
        c = rec.Candidate("S1", (), "")
        return rec.Scored(rec.Variant("GS", harmless, "t", "s", 0, c), "", flagged)

    assert rec.rate("x", [s(False, True), s(False, False)]).text == "1/2 = 50.0%"
    assert rec.rate("x", []).text == "n/a"
    assert s(True, True).outcome == "false_alarm"
    assert s(True, False).outcome == "passed"
