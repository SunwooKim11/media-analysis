"""I/O utilities – JSONL read/write to data/raw/."""

import json
from pathlib import Path

DATA_RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def save_jsonl(records: list[dict], filename: str) -> Path:
    """Append *records* to ``data/raw/<filename>.jsonl`` and return the path.

    Parameters
    ----------
    records : list[dict]
        Each dict is written as one JSON line.
    filename : str
        Stem name without extension (e.g. ``"naver_news"``).
    """
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_RAW_DIR / f"{filename}.jsonl"
    with open(path, "a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def load_jsonl(filename: str) -> list[dict]:
    """Read all records from ``data/raw/<filename>.jsonl``."""
    path = DATA_RAW_DIR / f"{filename}.jsonl"
    records: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
