"""PostgreSQL persistence (install the ``db`` extra). The only package that contains SQL."""

from boxing_ai.database.connection import DEFAULT_DSN, DSN_ENV, connect
from boxing_ai.database.migrate import migrate, sync_action_types
from boxing_ai.database.repository import (
    FighterInfo,
    ImportConflict,
    ImportResult,
    SourceInfo,
    VideoInfo,
    action_counts,
    audit,
    fighter_exists,
    get_video,
    import_annotation,
    import_source,
    list_fighters,
    list_sources,
    load_events,
    load_rounds,
    load_unobserved,
)

__all__ = [
    "DEFAULT_DSN",
    "DSN_ENV",
    "FighterInfo",
    "ImportConflict",
    "ImportResult",
    "SourceInfo",
    "VideoInfo",
    "action_counts",
    "audit",
    "connect",
    "fighter_exists",
    "get_video",
    "import_annotation",
    "import_source",
    "list_fighters",
    "list_sources",
    "load_events",
    "load_rounds",
    "load_unobserved",
    "migrate",
    "sync_action_types",
]
