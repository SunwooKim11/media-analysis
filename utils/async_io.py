"""Async I/O utilities -- streaming JSONL writer."""

import asyncio
import json
from pathlib import Path

import aiofiles

DATA_RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


class AsyncJsonlWriter:
    """Async-safe streaming JSONL writer.

    Parameters
    ----------
    filename : str
        Stem name without extension (e.g. ``"kookmin_press"``).
    batch_size : int
        Flush after this many buffered records. Default 5.
    """

    def __init__(self, filename: str, batch_size: int = 5):
        DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
        self.path = DATA_RAW_DIR / f"{filename}.jsonl"
        self.batch_size = batch_size
        self._lock = asyncio.Lock()
        self._buffer: list[str] = []
        self._count = 0

    async def write_record(self, record: dict) -> None:
        """Append a single record. Flushes when buffer reaches batch_size."""
        line = json.dumps(record, ensure_ascii=False) + "\n"
        async with self._lock:
            self._buffer.append(line)
            self._count += 1
            if len(self._buffer) >= self.batch_size:
                await self._flush()

    async def _flush(self) -> None:
        """Write buffered lines to disk. Caller must hold self._lock."""
        if not self._buffer:
            return
        async with aiofiles.open(self.path, "a", encoding="utf-8") as f:
            await f.write("".join(self._buffer))
        self._buffer.clear()

    async def close(self) -> None:
        """Flush remaining buffer."""
        async with self._lock:
            await self._flush()

    @property
    def count(self) -> int:
        return self._count
