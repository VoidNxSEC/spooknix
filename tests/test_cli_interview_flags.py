"""CLI interview: --list / --show / --diff agem antes do orchestrator."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from src import sessions_db
from src.cli import cli
from src.types import SessionRecord


@pytest.fixture
def populated_db(tmp_path: Path, monkeypatch):
    db_path = tmp_path / "sessions.db"
    monkeypatch.setattr(sessions_db, "_default_db_path", lambda: db_path)

    rubric_a = {
        "communication": {"score": 3, "comment": "ok"},
        "technical_depth": {"score": 4, "comment": "solid"},
        "confidence": {"score": 3, "comment": "fine"},
        "clarity": {"score": 4, "comment": "clear"},
        "examples": {"score": 2, "comment": "few"},
        "overall_comment": "decent",
    }
    rubric_b = {
        "communication": {"score": 4, "comment": "better"},
        "technical_depth": {"score": 4, "comment": "solid"},
        "confidence": {"score": 4, "comment": "fine"},
        "clarity": {"score": 5, "comment": "clear"},
        "examples": {"score": 3, "comment": "more"},
        "overall_comment": "improved",
    }

    a = SessionRecord.new(persona="sarah", scenario="behavioral", difficulty="standard")
    a.rubric_json = json.dumps(rubric_a)
    a.duration_s = 1500
    sa_id = sessions_db.insert(a, db_path=db_path)

    b = SessionRecord.new(persona="sarah", scenario="behavioral", difficulty="standard")
    b.rubric_json = json.dumps(rubric_b)
    b.duration_s = 1700
    sb_id = sessions_db.insert(b, db_path=db_path)

    return db_path, sa_id, sb_id


def test_interview_list_renderiza_tabela(populated_db):
    runner = CliRunner()
    result = runner.invoke(cli, ["interview", "--list"])
    assert result.exit_code == 0
    assert "sarah" in result.output
    assert "behavioral" in result.output


def test_interview_show_um_id(populated_db):
    _, sa_id, _ = populated_db
    runner = CliRunner()
    result = runner.invoke(cli, ["interview", "--show", str(sa_id)])
    assert result.exit_code == 0
    assert "sarah" in result.output
    assert "behavioral" in result.output


def test_interview_show_inexistente(populated_db):
    runner = CliRunner()
    result = runner.invoke(cli, ["interview", "--show", "9999"])
    assert result.exit_code == 0
    assert "não encontrada" in result.output


def test_interview_diff_renderiza_eixos(populated_db):
    _, sa_id, sb_id = populated_db
    runner = CliRunner()
    result = runner.invoke(cli, ["interview", "--diff", str(sa_id), str(sb_id)])
    assert result.exit_code == 0
    # delta de communication = 4-3 = +1
    assert "+1" in result.output
    assert "communication" in result.output
