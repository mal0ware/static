"""Read a few members of a large remote zip archive without downloading all of it.

A zip file keeps its index at the end, so ``zipfile`` only needs to seek and read. This
reader turns those reads into HTTP range requests through a ``fetch(first, last)``
callable, and fetches ahead in large chunks so each member costs a request or two.
"""

from __future__ import annotations

import zipfile
from collections.abc import Callable

Fetch = Callable[[int, int], bytes]


class RangeReader:
    """A seekable, read-only file whose bytes come from ``fetch(first, last)``.

    ``fetch`` returns the bytes from offset ``first`` to ``last`` inclusive. ``size`` is
    the length of the remote file. ``chunk`` is the least number of bytes asked for at
    a time.
    """

    def __init__(self, fetch: Fetch, size: int, chunk: int = 8 << 20) -> None:
        if size < 0 or chunk < 1:
            raise ValueError("size must not be negative and chunk must be positive")
        self._fetch = fetch
        self.size = size
        self.chunk = chunk
        self.requests = 0
        self._pos = 0
        self._start = 0
        self._buffer = b""

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            target = offset
        elif whence == 1:
            target = self._pos + offset
        elif whence == 2:
            target = self.size + offset
        else:
            raise ValueError("whence must be 0, 1 or 2")
        if target < 0:
            raise ValueError("cannot seek before the start")
        self._pos = target
        return self._pos

    def read(self, n: int = -1) -> bytes:
        if n < 0:
            n = self.size - self._pos
        n = min(n, self.size - self._pos)
        if n <= 0:
            return b""
        held = self._start <= self._pos and self._pos + n <= self._start + len(self._buffer)
        if not held:
            last = min(self._pos + max(n, self.chunk), self.size) - 1
            data = self._fetch(self._pos, last)
            if len(data) != last - self._pos + 1:
                raise OSError("the server returned a different number of bytes than asked for")
            self._buffer, self._start = data, self._pos
            self.requests += 1
        offset = self._pos - self._start
        self._pos += n
        return self._buffer[offset : offset + n]


def read_members(reader: RangeReader, names: list[str]) -> dict[str, bytes]:
    """The contents of ``names`` from the zip archive behind ``reader``."""
    with zipfile.ZipFile(reader) as archive:
        return {name: archive.read(name) for name in names}
