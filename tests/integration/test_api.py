"""HTTP API against a real database: hand-calculated numbers, errors, and video streaming with seeking."""

from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from boxing_ai.annotations import load_annotation  # noqa: E402
from boxing_ai.api import create_app  # noqa: E402
from boxing_ai.database import import_annotation, import_source  # noqa: E402

SAMPLE = Path(__file__).resolve().parents[2] / "data" / "annotations" / "sample-synthetic"


@pytest.fixture
def client(conn, test_dsn, tmp_path):
    annotation = load_annotation(SAMPLE)
    import_annotation(conn, annotation)
    return TestClient(create_app(dsn=test_dsn, media_dir=tmp_path)), annotation


def test_fighters_and_sources(client):
    c, _ = client
    assert [f["slug"] for f in c.get("/api/fighters").json()] == ["fighter-a", "fighter-b"]
    (source,) = c.get("/api/sources", params={"fighter": "fighter-a"}).json()
    assert (source["kind"], source["events"]) == ("HUMAN", 14)
    assert c.get("/api/sources", params={"fighter": "nobody"}).json() == []


def test_report_matches_hand_calculation(client):
    c, _ = client
    r = c.get("/api/fighters/fighter-a/report").json()
    assert (r["punches"], r["observed_ms"]) == (8, 350_000)
    assert r["punches_per_observed_minute"] == pytest.approx(8 / (350 / 60))
    assert r["outcomes"][0] == {
        "action_type": "JAB",
        "thrown": 5,
        "landed": 1,
        "blocked": 1,
        "missed": 2,
        "unknown": 1,
        "landed_rate": 0.25,
    }
    assert r["combos"]["2"][0] == {
        "ngram": ["JAB", "CROSS"],
        "count": 2,
        "known": 2,
        "landed": 1,
        "landed_rate": 0.5,
    }
    assert r["categories"] == ["DEFENSE", "FEINT", "PUNCH"] and r["gap_ms"] == 1000
    after_jab = {t["next"]: t["probability"] for t in r["transitions"] if t["context"] == ["JAB"]}
    assert after_jab["CROSS"] == pytest.approx(2 / 5)


def test_report_parameters(client):
    c, _ = client
    r = c.get("/api/fighters/fighter-a/report", params={"tokens": "commitment"}).json()
    assert [p["ngram"] for p in r["combos"]["3"]] == [
        ["CROSS", "SLIP", "JAB"],
        ["FEINT", "JAB_PROBE", "HOOK"],  # round 2, probe split out
        ["JAB", "CROSS", "SLIP"],
        ["JAB", "JAB", "CROSS"],
        ["SLIP", "JAB", "CROSS"],
    ]
    top2 = c.get("/api/fighters/fighter-a/report", params={"top": 2}).json()
    assert all(len(v) == 2 for v in top2["combos"].values())
    wide = c.get("/api/fighters/fighter-a/report", params={"gap_ms": 60_000}).json()
    assert wide["combos"]["2"][0]["count"] >= 2
    assert c.get("/api/fighters/fighter-a/report", params={"tokens": "nope"}).status_code == 422
    assert c.get("/api/fighters/fighter-a/report", params={"gap_ms": -1}).status_code == 422


def test_occurrences_locate_each_pattern_instance(client):
    c, _ = client
    occ = c.get("/api/fighters/fighter-a/occurrences", params={"ngram": ["JAB", "CROSS"]}).json()
    assert [(o["start_ms"], o["end_ms"], o["final_outcome"]) for o in occ] == [
        (14_710, 15_230, "BLOCKED"),
        (16_010, 16_580, "LANDED"),
    ]
    assert [e["action_type"] for e in occ[0]["events"]] == ["JAB", "CROSS"]
    assert c.get("/api/fighters/fighter-a/occurrences").status_code == 422  # ngram required


def test_errors_map_to_http_statuses(client, conn):
    c, annotation = client
    assert c.get("/api/fighters/nobody/report").status_code == 404
    assert c.get("/api/sources/999/timeline").status_code == 404

    second = annotation.meta.model_copy(
        update={"annotator": annotation.meta.annotator.model_copy(update={"name": "other"})}
    )
    import_source(conn, second, annotation.events)
    response = c.get("/api/fighters/fighter-a/report")
    assert response.status_code == 409 and "pick one with --source" in response.json()["detail"]
    assert c.get("/api/fighters/fighter-a/report", params={"source": "other"}).status_code == 200


def test_timeline_and_video_streaming(client, conn, tmp_path):
    c, annotation = client
    t = c.get("/api/sources/1/timeline").json()
    assert t["video_url"] is None and len(t["events"]) == 14
    assert [r["number"] for r in t["rounds"]] == [1, 2] and t["unobserved"][0]["kind"] == "REPLAY"

    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "clip.mp4").write_bytes(bytes(range(256)) * 4)
    conn.execute("UPDATE videos SET path = 'raw/clip.mp4'")
    conn.commit()
    t = c.get("/api/sources/1/timeline").json()
    assert t["video_url"] == "/api/videos/sample-synthetic/file"

    full = c.get(t["video_url"])
    assert full.status_code == 200 and len(full.content) == 1024
    part = c.get(t["video_url"], headers={"Range": "bytes=256-511"})
    assert part.status_code == 206 and part.content == bytes(range(256))
    assert part.headers["content-range"] == "bytes 256-511/1024"


@pytest.mark.parametrize(
    ("path", "status"), [("../outside.mp4", 403), ("raw/missing.mp4", 404), (None, 404)]
)
def test_video_path_safety(client, conn, tmp_path, path, status):
    c, _ = client
    (tmp_path.parent / "outside.mp4").write_bytes(b"secret")
    conn.execute("UPDATE videos SET path = %s", (path,))
    conn.commit()
    assert c.get("/api/videos/sample-synthetic/file").status_code == status
