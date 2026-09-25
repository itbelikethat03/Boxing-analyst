"""Minimal migration runner: numbered ``migrations/*.sql`` files, applied once each, recorded in
``schema_migrations``. After migrating, ``action_types`` is synced from ``ontology.py`` (the code is the source of
truth; the table exists so events get a foreign key to it).
"""

from __future__ import annotations

from pathlib import Path

import psycopg

from boxing_ai.ontology import ACTIONS

MIGRATIONS_DIR = Path(__file__).with_name("migrations")


def migrate(conn: psycopg.Connection) -> list[str]:
    """Apply pending migrations and sync action types in one transaction; return the versions applied now."""
    applied: list[str] = []
    with conn.transaction():
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            " version TEXT PRIMARY KEY, applied_at TIMESTAMPTZ NOT NULL DEFAULT now())"
        )
        conn.execute("SELECT pg_advisory_xact_lock(hashtext('boxing_ai.migrate'))")
        done = {row[0] for row in conn.execute("SELECT version FROM schema_migrations")}
        for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
            if path.stem in done:
                continue
            # Executed without parameters, so one call may hold many statements.
            conn.execute(path.read_text(encoding="utf-8"))
            conn.execute("INSERT INTO schema_migrations (version) VALUES (%s)", (path.stem,))
            applied.append(path.stem)
        sync_action_types(conn)
    return applied


def sync_action_types(conn: psycopg.Connection) -> None:
    """Make ``action_types`` equal to the ontology. Fails (FK) rather than orphan stored events."""
    rows = [(spec.code, spec.category.value) for spec in ACTIONS.values()]
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO action_types (code, category) VALUES (%s, %s)"
            " ON CONFLICT (code) DO UPDATE SET category = EXCLUDED.category",
            rows,
        )
        cur.execute("DELETE FROM action_types WHERE code <> ALL(%s)", ([code for code, _ in rows],))
