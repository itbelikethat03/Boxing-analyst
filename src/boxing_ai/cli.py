"""``boxing-ai`` command line: init-db | import | sources | report | db-check | serve. Parses arguments and renders text;
all logic lives in ``annotations``, ``database``, ``reporting`` and the analytics they call.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from boxing_ai.annotations import AnnotationError, load_annotation
from boxing_ai.annotations.loader import META_FILE
from boxing_ai.database import DSN_ENV, audit, connect, import_annotation, list_sources, migrate
from boxing_ai.reporting import AmbiguousSource, Report, fighter_report
from boxing_ai.sequences import END, START, TOKENIZERS, StreamSpec


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return args.run(args)
    except (
        LookupError,
        ValueError,
    ) as exc:  # AmbiguousSource, unknown fighter, ImportConflict, ...
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="boxing-ai", description="Boxing event analytics.")
    parser.add_argument("--dsn", help=f"PostgreSQL DSN (default: ${DSN_ENV} or the local cluster)")
    sub = parser.add_subparsers(required=True, metavar="command")

    p = sub.add_parser("init-db", help="create/upgrade the schema and sync action types")
    p.set_defaults(run=_init_db)

    p = sub.add_parser("import", help="validate and import annotation folders")
    p.add_argument("paths", nargs="+", type=Path, help="annotation folder(s), or a folder of them")
    p.set_defaults(run=_import)

    p = sub.add_parser("sources", help="list imported sources")
    p.add_argument("--fight")
    p.add_argument("--fighter")
    p.set_defaults(run=_sources)

    p = sub.add_parser("report", help="statistics and patterns for one fighter")
    p.add_argument("fighter", help="fighter slug, e.g. fighter-a")
    p.add_argument("--fight", help="only this fight (default: all of the fighter's fights)")
    p.add_argument("--source", help="source name or name@version when a video has several")
    p.add_argument("--tokens", choices=TOKENIZERS, default="action", help="how actions are named")
    p.add_argument("--gap-ms", type=int, default=StreamSpec().gap_ms, help="burst gap (ms)")
    p.add_argument("--min-count", type=int, default=1, help="hide patterns seen fewer times")
    p.add_argument("--top", type=int, default=10, help="rows per pattern table")
    p.set_defaults(run=_report)

    p = sub.add_parser("db-check", help="audit stored data for inconsistencies")
    p.set_defaults(run=_db_check)

    p = sub.add_parser("serve", help="run the HTTP API for the web front-end (needs the api extra)")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true", help="restart on code changes (development)")
    p.set_defaults(run=_serve)
    return parser


# --- commands --------------------------------------------------------------------------------------


def _init_db(args: argparse.Namespace) -> int:
    with connect(args.dsn) as conn:
        applied = migrate(conn)
    print(f"applied: {', '.join(applied)}" if applied else "schema up to date")
    return 0


def _import(args: argparse.Namespace) -> int:
    folders = [f for path in args.paths for f in _annotation_folders(path)]
    if not folders:
        print(f"error: no folder with {META_FILE} found", file=sys.stderr)
        return 1
    failed = 0
    with connect(args.dsn) as conn:
        migrate(conn)
        for folder in folders:
            try:
                result = import_annotation(conn, load_annotation(folder))
            except AnnotationError as exc:
                failed += 1
                print(exc, file=sys.stderr)
                continue
            except ValueError as exc:  # ImportConflict
                failed += 1
                print(f"{folder}: {exc}", file=sys.stderr)
                continue
            print(f"{folder}: {result.status} (source {result.source_id}, {result.events} events)")
    return 1 if failed else 0


def _annotation_folders(path: Path) -> list[Path]:
    if (path / META_FILE).is_file():
        return [path]
    return sorted(p for p in path.iterdir() if (p / META_FILE).is_file()) if path.is_dir() else []


def _sources(args: argparse.Namespace) -> int:
    with connect(args.dsn) as conn:
        rows = list_sources(conn, fight=args.fight, fighter=args.fighter)
    for s in rows:
        print(f"{s.id:>5}  {s.fight}  {s.video}  {s.kind} {s.name}@{s.version}  {s.events} events")
    if not rows:
        print("no sources")
    return 0


def _report(args: argparse.Namespace) -> int:
    spec = StreamSpec(tokenizer=TOKENIZERS[args.tokens], gap_ms=args.gap_ms)
    with connect(args.dsn) as conn:
        try:
            report = fighter_report(
                conn,
                args.fighter,
                fight=args.fight,
                source=args.source,
                spec=spec,
                min_count=args.min_count,
                limit=args.top,
            )
        except AmbiguousSource as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    print(render_report(report, tokens=args.tokens))
    return 0


def _db_check(args: argparse.Namespace) -> int:
    with connect(args.dsn) as conn:
        problems = audit(conn)
    for p in problems:
        print(f"PROBLEM: {p}")
    print("ok" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


def _serve(args: argparse.Namespace) -> int:
    import os

    import uvicorn  # optional dependency: only this command needs it

    if args.dsn:
        os.environ[DSN_ENV] = args.dsn  # the app reads it at startup
    # Local only: the API has no authentication.
    uvicorn.run("boxing_ai.api:app", host="127.0.0.1", port=args.port, reload=args.reload)
    return 0


# --- rendering -------------------------------------------------------------------------------------


def _pct(value: float | None) -> str:
    return "  -" if value is None else f"{value:.0%}"


def _mmss(ms: int) -> str:
    return f"{ms // 60_000}:{ms // 1000 % 60:02d}"


def _gram(tokens: Sequence[str]) -> str:
    return " > ".join(tokens)


def render_report(report: Report, *, tokens: str = "action") -> str:
    r = report
    categories = "+".join(sorted(c.value for c in r.spec.categories))
    lines = [f"Fighter: {r.fighter}", "Sources:"]
    lines += [f"  {s.fight} / {s.video}: {s.name}@{s.version} ({s.kind})" for s in r.sources]
    lines.append(f"Stream: {categories}, tokens={tokens}, burst gap {r.spec.gap_ms} ms")
    rate = r.punches_per_observed_minute
    lines.append(
        f"\nPunches: {r.punches} in {_mmss(r.observed_ms)} observed"
        + (f" = {rate:.2f} per minute" if rate is not None else "")
    )

    lines.append(
        f"\n{'Punch outcomes':<16}{'thrown':>7}{'landed':>8}{'blocked':>8}{'missed':>8}"
        f"{'unknown':>8}{'landed%':>9}"
    )
    for o in r.outcomes:
        lines.append(
            f"  {o.action_type:<14}{o.thrown:>7}{o.landed:>8}{o.blocked:>8}{o.missed:>8}"
            f"{o.unknown:>8}{_pct(o.landed_rate):>9}"
        )
    lines.append("  (landed% = landed / punches with a known outcome)")

    lines.append("\nPer round")
    for rs in r.rounds:
        per_min = rs.per_observed_minute
        lines.append(
            f"  {rs.video} R{rs.round_number:<3}{rs.count:>4} punches  {_mmss(rs.observed_ms)} observed"
            + (f"  {per_min:.2f}/min" if per_min is not None else "")
        )

    for n, patterns in r.combos.items():
        lines.append(f"\nTop {n}-action sequences{'':<16}{'count':>6}{'final punch landed':>21}")
        for p in patterns:
            landed = "-" if p.landed_rate is None else f"{_pct(p.landed_rate)} of {p.known}"
            lines.append(f"  {_gram(p.ngram):<38}{p.count:>6}{landed:>21}")
        if not patterns:
            lines.append("  (none)")

    lines.append(
        "\nBursts open with:  " + ", ".join(f"{_gram(e.ngram)} {e.count}" for e in r.entries)
    )
    lines.append("Bursts close with: " + ", ".join(f"{_gram(e.ngram)} {e.count}" for e in r.exits))

    lines.append("\nWhat follows (P(next | action), count/total)")
    grouped = defaultdict(list)
    for t in r.transitions:
        grouped[t.context].append(t)
    for context, rows in sorted(grouped.items(), key=lambda kv: (-kv[1][0].total, kv[0])):
        if context == (START,):
            continue
        best = sorted(rows, key=lambda t: (-t.count, t.next))[:3]
        shown = ", ".join(
            f"{'end' if t.next == END else t.next} {t.probability:.0%} ({t.count}/{t.total})"
            for t in best
        )
        lines.append(f"  {_gram(context):<12} -> {shown}")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
