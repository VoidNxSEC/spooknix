"""Schema validation para personas/*.yaml + loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from src import personas


def test_list_personas_inclui_sarah_marcus_priya():
    names = personas.list_personas()
    assert "sarah" in names
    assert "marcus" in names
    assert "priya" in names


def test_load_persona_sarah():
    p = personas.load_persona("sarah")
    assert p.name == "Sarah"
    assert "Recruiter" in p.system_prompt
    assert p.voice_ref_text


def test_load_persona_inexistente_levanta(tmp_path: Path):
    with pytest.raises(personas.PersonaNotFound):
        personas.load_persona("nao_existe", root=tmp_path)


def test_load_persona_yaml_invalido(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("language: en\n", encoding="utf-8")  # falta name + system_prompt
    with pytest.raises(ValueError):
        personas.load_persona("bad", root=tmp_path)


def test_todos_yamls_carregam_sem_erro():
    """Smoke test sobre todos os arquivos do diretório real."""
    for name in personas.list_personas():
        p = personas.load_persona(name)
        assert p.name
        assert p.system_prompt
