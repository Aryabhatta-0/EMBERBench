"""Canonical JSON encoding used for hashing, comparison and storage.

Two equal values always encode to the same bytes: keys are sorted, there is
no insignificant whitespace, output is ASCII and NaN/Infinity are rejected.
"""

import hashlib
import json
from typing import Any


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def canonical_bytes(obj: Any) -> bytes:
    return canonical_json(obj).encode("ascii")


def sha256_hex(obj: Any) -> str:
    return hashlib.sha256(canonical_bytes(obj)).hexdigest()
