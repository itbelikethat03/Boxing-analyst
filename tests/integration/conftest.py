"""Integration fixtures: a throw-away database per test session, emptied before every test.

``BOXING_AI_TEST_DSN`` must point at a server where the user may CREATE DATABASE, e.g.
``postgresql://postgres@localhost:5432/postgres``. The tests create and drop ``boxing_ai_pytest`` there and never
touch any other database. Without the variable (or without psycopg) every integration test is skipped.
"""

from __future__ import annotations

import os

import pytest

psycopg = pytest.importorskip("psycopg")
from psycopg.conninfo import make_conninfo  # noqa: E402

from boxing_ai.database import migrate  # noqa: E402

TEST_DB = "boxing_ai_pytest"
DATA_TABLES = "events, unobserved_intervals, event_sources, rounds, videos, fight_participants, fights, fighters"


def pytest_collection_modifyitems(items):
    for item in items:
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)


@pytest.fixture(scope="session")
def test_dsn():
    admin_dsn = os.environ.get("BOXING_AI_TEST_DSN")
    if not admin_dsn:
        pytest.skip("BOXING_AI_TEST_DSN is not set")
    with psycopg.connect(admin_dsn, autocommit=True) as admin:
        admin.execute(f"DROP DATABASE IF EXISTS {TEST_DB} WITH (FORCE)")
        admin.execute(f"CREATE DATABASE {TEST_DB}")
    dsn = make_conninfo(admin_dsn, dbname=TEST_DB)
    with psycopg.connect(dsn) as conn:
        migrate(conn)
    yield dsn
    with psycopg.connect(admin_dsn, autocommit=True) as admin:
        admin.execute(f"DROP DATABASE IF EXISTS {TEST_DB} WITH (FORCE)")


@pytest.fixture
def conn(test_dsn):
    with psycopg.connect(test_dsn) as conn:
        conn.execute(f"TRUNCATE {DATA_TABLES} RESTART IDENTITY CASCADE")
        conn.commit()
        yield conn
