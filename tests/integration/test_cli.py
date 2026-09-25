"""End to end: `boxing-ai import` the sample folder, then `report` must show the hand-calculated numbers."""

from pathlib import Path

from boxing_ai.cli import main

ANNOTATIONS = Path(__file__).resolve().parents[2] / "data" / "annotations"


def test_import_report_and_check(conn, test_dsn, capsys):
    assert main(["--dsn", test_dsn, "import", str(ANNOTATIONS)]) == 0
    assert "sample-synthetic: created (source 1, 14 events)" in capsys.readouterr().out
    assert main(["--dsn", test_dsn, "import", str(ANNOTATIONS / "sample-synthetic")]) == 0
    assert "unchanged" in capsys.readouterr().out

    assert main(["--dsn", test_dsn, "report", "fighter-a"]) == 0
    out = capsys.readouterr().out
    for expected in [
        "Punches: 8 in 5:50 observed = 1.37 per minute",  # 8 / (350 s / 60)
        "  JAB                 5       1       1       2       1      25%",
        "  CROSS               2       1       1       0       0      50%",
        "  sample-synthetic R1     6 punches  2:50 observed  2.12/min",  # 6 / (170 s / 60)
        "  JAB > CROSS                                2             50% of 2",
        "  FEINT > JAB > HOOK                         1            100% of 1",
        "Bursts open with:  JAB 2, FEINT 1",
        "  JAB          -> CROSS 40% (2/5), end 20% (1/5), HOOK 20% (1/5)",
    ]:
        assert expected in out, expected

    assert main(["--dsn", test_dsn, "db-check"]) == 0
    assert capsys.readouterr().out.strip() == "ok"


def test_report_errors(conn, test_dsn, capsys):
    assert main(["--dsn", test_dsn, "report", "nobody"]) == 1
    assert "unknown fighter 'nobody'" in capsys.readouterr().err


def test_invalid_folder_is_reported_and_fails(conn, test_dsn, tmp_path, capsys):
    (tmp_path / "meta.toml").write_text("schema_version = 1\n", encoding="utf-8")
    assert main(["--dsn", test_dsn, "import", str(tmp_path)]) == 1
    assert "meta.toml: annotator: Field required" in capsys.readouterr().err
