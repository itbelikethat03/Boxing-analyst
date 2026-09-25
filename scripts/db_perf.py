"""Measure the primary queries on a synthetic ~2M-event database (docs/plan.md §3 verification target).

    .venv\\Scripts\\python.exe scripts\\db_perf.py [--admin-dsn DSN] [--keep]

Creates the database ``boxing_ai_perf`` (dropped afterwards unless --keep), fills it with 200 fighters, 1000 fights
x 2000 events, then prints EXPLAIN ANALYZE timings for: one fight's timeline, one fighter's events across all of
their fights, and a per-fighter multi-fight aggregation. Targets: timeline <= tens of ms, aggregation <= 1 s.
"""

from __future__ import annotations

import argparse
import time

import psycopg
from psycopg.conninfo import make_conninfo

from boxing_ai.database import migrate
from boxing_ai.database.repository import load_events

PERF_DB = "boxing_ai_perf"
FIGHTERS, FIGHTS, EVENTS_PER_FIGHT = 200, 1000, 2000

FILL = f"""
INSERT INTO fighters (slug, name, stance)
SELECT 'fighter-' || i, 'Fighter ' || i, 'ORTHODOX' FROM generate_series(1, {FIGHTERS}) i;

INSERT INTO fights (slug, kind) SELECT 'fight-' || i, 'FIGHT' FROM generate_series(1, {FIGHTS}) i;

INSERT INTO fight_participants (fight_id, fighter_id, corner)
SELECT i, (i % {FIGHTERS}) + 1, 'RED' FROM generate_series(1, {FIGHTS}) i
UNION ALL
SELECT i, ((i + 1) % {FIGHTERS}) + 1, 'BLUE' FROM generate_series(1, {FIGHTS}) i;

INSERT INTO videos (fight_id, slug, duration_ms) SELECT i, 'video-' || i, 3600000
FROM generate_series(1, {FIGHTS}) i;

INSERT INTO rounds (video_id, round_number, start_ms, end_ms)
SELECT v, r, (r - 1) * 240000, (r - 1) * 240000 + 180000
FROM generate_series(1, {FIGHTS}) v, generate_series(1, 12) r;

INSERT INTO event_sources (video_id, kind, name, ontology_version)
SELECT i, 'HUMAN', 'perf', '0.2' FROM generate_series(1, {FIGHTS}) i;

-- 2000 events per fight: alternating fighters, JAB/CROSS/HOOK, spread over the 12 rounds (~7 s apart).
INSERT INTO events (source_id, video_id, fight_id, fighter_id, round_number, start_ms, end_ms,
                    action_type, category, side, outcome)
SELECT f, f, f,
       CASE WHEN n % 2 = 0 THEN (f % {FIGHTERS}) + 1 ELSE ((f + 1) % {FIGHTERS}) + 1 END,
       r, (r - 1) * 240000 + k * 1000, (r - 1) * 240000 + k * 1000 + 200,
       (ARRAY['JAB', 'CROSS', 'HOOK'])[n % 3 + 1], 'PUNCH',
       (ARRAY['LEAD', 'REAR', 'LEAD'])[n % 3 + 1],
       (ARRAY['LANDED', 'BLOCKED', 'MISSED'])[n % 3 + 1]
FROM generate_series(1, {FIGHTS}) f,
     generate_series(0, {EVENTS_PER_FIGHT} - 1) n,
     LATERAL (SELECT n % 12 + 1 AS r, n / 12 AS k) pos;
ANALYZE;
"""

QUERIES = {
    "one fight timeline (2000 events)": (
        "SELECT * FROM events WHERE source_id = %s ORDER BY start_ms, end_ms, id",
        (500,),
    ),
    "one fighter, all their fights (10 fights, ~10k events)": (
        "SELECT * FROM events WHERE fighter_id = %s ORDER BY fight_id, start_ms",
        (42,),
    ),
    "per-fighter action totals across fights": (
        "SELECT action_type, outcome, count(*) FROM events WHERE fighter_id = %s"
        " GROUP BY action_type, outcome",
        (42,),
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--admin-dsn", default="postgresql://postgres@localhost:5432/postgres")
    parser.add_argument("--keep", action="store_true", help="keep the perf database afterwards")
    args = parser.parse_args()

    with psycopg.connect(args.admin_dsn, autocommit=True) as admin:
        admin.execute(f"DROP DATABASE IF EXISTS {PERF_DB} WITH (FORCE)")
        admin.execute(f"CREATE DATABASE {PERF_DB}")
    dsn = make_conninfo(args.admin_dsn, dbname=PERF_DB)
    try:
        with psycopg.connect(dsn) as conn:
            migrate(conn)
            t0 = time.perf_counter()
            conn.execute(FILL)
            conn.commit()
            (count,) = conn.execute("SELECT count(*) FROM events").fetchone()
            print(f"filled {count:,} events in {time.perf_counter() - t0:.1f} s\n")

            for label, (sql, params) in QUERIES.items():
                plan = conn.execute(f"EXPLAIN (ANALYZE, BUFFERS) {sql}", params).fetchall()
                lines = [row[0] for row in plan]
                execution = next(line for line in lines if line.startswith("Execution Time"))
                scan = next(line.strip() for line in lines if "Scan" in line)
                print(f"{label}: {execution}\n    {scan}")

            t0 = time.perf_counter()
            events = load_events(conn, [500])
            elapsed = (time.perf_counter() - t0) * 1000
            print(f"\nrepository.load_events -> {len(events)} Event objects in {elapsed:.0f} ms")
    finally:
        if not args.keep:
            with psycopg.connect(args.admin_dsn, autocommit=True) as admin:
                admin.execute(f"DROP DATABASE IF EXISTS {PERF_DB} WITH (FORCE)")


if __name__ == "__main__":
    main()
