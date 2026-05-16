"""Schema validation para scenarios/*.yaml + loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from src import scenarios


def test_list_scenarios_inclui_principais():
    names = scenarios.list_scenarios()
    for n in ("behavioral", "system_design", "frontend", "backend", "ml", "leadership", "product"):
        assert n in names, f"cenário {n} ausente"


def test_load_scenario_system_design_standard():
    cfg = scenarios.load_scenario("system_design", "standard")
    assert cfg.scenario.interview_type == "System Design"
    assert cfg.scenario.difficulty == "standard"
    assert cfg.scenario.duration_mins == 45
    assert "technical_depth" in cfg.rubric_weights
    assert sum(cfg.rubric_weights.values()) == pytest.approx(1.0, abs=0.01)


def test_load_scenario_difficulty_invalido():
    with pytest.raises(ValueError):
        scenarios.load_scenario("behavioral", "impossivel")


def test_load_scenario_nao_existente(tmp_path: Path):
    with pytest.raises(scenarios.ScenarioNotFound):
        scenarios.load_scenario("nao_existe", "easy", root=tmp_path)


def test_todos_os_niveis_carregam():
    for name in scenarios.list_scenarios():
        for diff in ("easy", "standard", "hard"):
            cfg = scenarios.load_scenario(name, diff)
            assert cfg.scenario.duration_mins > 0
            assert isinstance(cfg.rubric_weights, dict)
