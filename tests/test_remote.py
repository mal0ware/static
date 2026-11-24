"""Reading members of a zip archive through range requests, against local bytes."""

from __future__ import annotations

import io
import zipfile

import numpy as np
import pytest

from static.remote import RangeReader, read_members


def archive() -> bytes:
    rng = np.random.default_rng(0)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as out:
        out.writestr("pack/big.bin", rng.bytes(200_000))
        out.writestr("pack/a.csv", "x,y\n1,2\n")
        out.writestr("pack/other.bin", rng.bytes(300_000))
        out.writestr("pack/b.txt", "hello")
    return buffer.getvalue()


def test_reads_only_the_members_asked_for() -> None:
    data = archive()
    asked: list[tuple[int, int]] = []

    def fetch(first: int, last: int) -> bytes:
        asked.append((first, last))
        return data[first : last + 1]

    reader = RangeReader(fetch, len(data), chunk=4096)
    members = read_members(reader, ["pack/a.csv", "pack/b.txt"])
    assert members == {"pack/a.csv": b"x,y\n1,2\n", "pack/b.txt": b"hello"}
    assert reader.requests == len(asked)
    assert sum(last - first + 1 for first, last in asked) < len(data) // 10


def test_large_member_round_trips() -> None:
    data = archive()
    reader = RangeReader(lambda a, b: data[a : b + 1], len(data), chunk=1 << 20)
    with zipfile.ZipFile(io.BytesIO(data)) as local:
        expected = local.read("pack/other.bin")
    assert read_members(reader, ["pack/other.bin"])["pack/other.bin"] == expected


def test_seek_tell_and_read_past_the_end() -> None:
    data = bytes(range(256))
    reader = RangeReader(lambda a, b: data[a : b + 1], len(data), chunk=16)
    assert reader.seekable() and reader.readable()
    assert reader.seek(-6, 2) == 250
    assert reader.read(100) == data[250:]
    assert reader.read(1) == b""
    reader.seek(10)
    assert reader.read(4) == data[10:14]
    assert reader.seek(2, 1) == 16
    assert reader.tell() == 16
    assert reader.read() == data[16:]


def test_reads_inside_the_held_chunk_cost_no_request() -> None:
    data = bytes(1000)
    reader = RangeReader(lambda a, b: data[a : b + 1], len(data), chunk=500)
    reader.read(10)
    reader.read(400)
    assert reader.requests == 1
    reader.read(200)
    assert reader.requests == 2


def test_bad_arguments_and_short_answers() -> None:
    with pytest.raises(ValueError, match="size"):
        RangeReader(lambda a, b: b"", -1)
    reader = RangeReader(lambda a, b: b"", 10)
    with pytest.raises(OSError, match="different number"):
        reader.read(4)
    with pytest.raises(ValueError, match="whence"):
        reader.seek(0, 3)
    with pytest.raises(ValueError, match="before the start"):
        reader.seek(-1)
