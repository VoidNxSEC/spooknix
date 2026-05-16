"""Parser tolerante para Rubric do evaluator."""

from __future__ import annotations

from src.rubric import Rubric, Axis, parse_rubric


def test_parse_json_cru():
    text = (
        '{"communication": {"score": 4, "comment": "clear"},'
        ' "technical_depth": {"score": 3, "comment": "ok"},'
        ' "confidence": {"score": 5, "comment": "great"},'
        ' "clarity": {"score": 4, "comment": "good"},'
        ' "examples": {"score": 2, "comment": "few"},'
        ' "overall_comment": "decent"}'
    )
    r = parse_rubric(text)
    assert r.communication.score == 4
    assert r.examples.comment == "few"
    assert r.overall_comment == "decent"


def test_parse_json_fenceado_com_prosa():
    text = (
        "Sure, here is the rubric:\n\n"
        "```json\n"
        '{"communication": {"score": 3, "comment": "ok"}}\n'
        "```\n\n"
        "Hope that helps!"
    )
    r = parse_rubric(text)
    assert r.communication.score == 3
    assert r.technical_depth.score == 0  # default


def test_parse_score_inteiro_solto():
    """LLM ocasionalmente devolve só o número."""
    text = '{"clarity": 4, "examples": 2}'
    r = parse_rubric(text)
    assert r.clarity.score == 4
    assert r.examples.score == 2


def test_parse_payload_invalido_da_default():
    r = parse_rubric("no JSON whatsoever, just rambling")
    assert isinstance(r, Rubric)
    assert r.communication.score == 0


def test_parse_string_vazia():
    r = parse_rubric("")
    assert r == Rubric()


def test_weighted_score_uniforme():
    r = Rubric(
        communication=Axis(score=4),
        technical_depth=Axis(score=4),
        confidence=Axis(score=4),
        clarity=Axis(score=4),
        examples=Axis(score=4),
    )
    assert r.weighted_score() == 4.0


def test_weighted_score_com_pesos():
    r = Rubric(
        communication=Axis(score=2),
        technical_depth=Axis(score=5),
        confidence=Axis(score=0),
        clarity=Axis(score=0),
        examples=Axis(score=0),
    )
    weights = {"communication": 0, "technical_depth": 1.0, "confidence": 0, "clarity": 0, "examples": 0}
    assert r.weighted_score(weights) == 5.0
