"""Annotation folders (``meta.toml`` + ``events.csv``) <-> validated Events. Format: ``docs/annotation.md``."""

from boxing_ai.annotations.loader import (
    Annotation,
    AnnotationError,
    AnnotationProblem,
    context_of,
    load_annotation,
)
from boxing_ai.annotations.meta import AnnotationMeta, SourceKind
from boxing_ai.annotations.timecodes import format_timecode, parse_timecode
from boxing_ai.annotations.writer import write_events_csv

__all__ = [
    "Annotation",
    "AnnotationError",
    "AnnotationMeta",
    "AnnotationProblem",
    "SourceKind",
    "context_of",
    "format_timecode",
    "load_annotation",
    "parse_timecode",
    "write_events_csv",
]
