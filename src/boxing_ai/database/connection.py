"""Connecting to PostgreSQL. The DSN comes from ``BOXING_AI_DSN``, defaulting to the local cluster of scripts/pg.ps1."""

from __future__ import annotations

import os

import psycopg

DSN_ENV = "BOXING_AI_DSN"
DEFAULT_DSN = "postgresql://postgres@localhost:5432/boxing_ai"


def connect(dsn: str | None = None) -> psycopg.Connection:
    return psycopg.connect(dsn or os.environ.get(DSN_ENV, DEFAULT_DSN))
