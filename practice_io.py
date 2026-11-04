"""Small output helpers shared by the two independent practice programs."""

import csv
import hashlib
import json
from pathlib import Path


def new_directory(path):
    path = Path(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"Choose a new or empty output directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def write_csv(path, rows, fields=None):
    if fields is None:
        if not rows:
            raise ValueError("Cannot infer CSV fields from no rows")
        fields = list(rows[0])
    with Path(path).open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def strict_json(path):
    def reject(value):
        raise ValueError(f"Non-finite JSON number: {value}")

    with Path(path).open(encoding="utf-8-sig") as handle:
        return json.load(handle, parse_constant=reject)

