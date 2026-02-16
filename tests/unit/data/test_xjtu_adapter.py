"""Tests for XJTU-SY archive adapters."""

from __future__ import annotations

from pathlib import Path

import pytest

from whirdetective.data.adapters.xjtu_sy import iter_xjtu_csv_payloads, list_xjtu_parts


def test_list_xjtu_parts_lists_part_archives(tmp_path: Path) -> None:
    (tmp_path / "XJTU-SY_Bearing_Datasets.part01.rar").write_text("x", encoding="utf-8")
    (tmp_path / "XJTU-SY_Bearing_Datasets.part02.rar").write_text("x", encoding="utf-8")
    (tmp_path / "ignore.txt").write_text("x", encoding="utf-8")

    parts = list_xjtu_parts(tmp_path)
    assert [part.name for part in parts] == [
        "XJTU-SY_Bearing_Datasets.part01.rar",
        "XJTU-SY_Bearing_Datasets.part02.rar",
    ]


def test_iter_xjtu_csv_payloads_filters_non_csv_entries(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "XJTU-SY_Bearing_Datasets.part06.rar"
    archive_path.write_text("placeholder", encoding="utf-8")

    class _Entry:
        def __init__(self, pathname: str, payload: bytes) -> None:
            self.pathname = pathname
            self._payload = payload

        def get_blocks(self):
            return (self._payload,)

    class _FakeReader:
        def __enter__(self) -> "_FakeReader":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def __iter__(self):
            return iter(
                (
                    _Entry(
                        "XJTU-SY_Bearing_Datasets/40Hz10kN/Bearing3_4/1106.csv",
                        b"Horizontal_vibration_signals,Vertical_vibration_signals\n1.0,2.0\n",
                    ),
                    _Entry("XJTU-SY_Bearing_Datasets/40Hz10kN/Bearing3_4/readme.txt", b"x"),
                )
            )

    monkeypatch.setattr(
        "whirdetective.data.adapters.xjtu_sy.libarchive.file_reader",
        lambda _: _FakeReader(),
    )
    payloads = iter_xjtu_csv_payloads(archive_path)
    assert len(payloads) == 1
    entry, payload = payloads[0]
    assert entry.condition == "40Hz10kN"
    assert entry.bearing_id == "Bearing3_4"
    assert entry.snapshot_index == 1106
    assert payload.startswith(b"Horizontal_vibration_signals")
