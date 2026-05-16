"""Roundtrip SQLite para SessionRecord."""

from __future__ import annotations

from pathlib import Path

import pytest

from src import sessions_db
from src.types import SessionRecord


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "sessions.db"


def test_insert_e_get_roundtrip(db_path: Path):
    rec = SessionRecord.new(persona="sarah", scenario="behavioral", difficulty="hard")
    rec.duration_s = 123.4
    rec.transcript_path = "/tmp/t.md"
    rec.rubric_json = '{"clarity":{"score":4,"comment":"clean"}}'

    sid = sessions_db.insert(rec, db_path=db_path)
    assert sid > 0

    fetched = sessions_db.get(sid, db_path=db_path)
    assert fetched is not None
    assert fetched.persona == "sarah"
    assert fetched.scenario == "behavioral"
    assert fetched.duration_s == pytest.approx(123.4)


def test_list_all_ordena_por_ts_desc(db_path: Path):
    r1 = SessionRecord.new(persona="a", scenario="b", difficulty="easy")
    r1.ts = "2026-05-10T10:00:00"
    r2 = SessionRecord.new(persona="a", scenario="b", difficulty="easy")
    r2.ts = "2026-05-12T10:00:00"
    sessions_db.insert(r1, db_path=db_path)
    sessions_db.insert(r2, db_path=db_path)

    rows = sessions_db.list_all(db_path=db_path)
    assert rows[0].ts == "2026-05-12T10:00:00"
    assert rows[1].ts == "2026-05-10T10:00:00"


def test_rubric_dict_parser_json_invalido(db_path: Path):
    rec = SessionRecord.new(persona="x", scenario="y", difficulty="easy")
    rec.rubric_json = "not json"
    sid = sessions_db.insert(rec, db_path=db_path)
    fetched = sessions_db.get(sid, db_path=db_path)
    assert sessions_db.rubric_dict(fetched) == {}


def test_get_inexistente_retorna_none(db_path: Path):
    assert sessions_db.get(999, db_path=db_path) is None


def test_migrate_idempotente(db_path: Path):
    """Reabrir o DB não duplica tabelas nem perde dados."""
    rec = SessionRecord.new(persona="a", scenario="b", difficulty="easy")
    sid = sessions_db.insert(rec, db_path=db_path)
    # Reabre
    fetched = sessions_db.get(sid, db_path=db_path)
    assert fetched is not None
