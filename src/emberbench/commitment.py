"""Salted commitments to frozen evaluation suites (docs/harness-protocol.md, section 5).

An evaluator publishes ``suite_commitment(pack_digest, salt)`` before any
evaluation and keeps the environment pack and the salt private. The pack
digest already binds every file of the suite and its seed schedule; the
256-bit salt hides it, so the public commitment reveals nothing about the
suite, and a later reveal of the pack and the salt lets anyone check that the
scored suite is the committed one. Rotating the suite publishes a new
commitment; the scoring contract does not change.
"""

from __future__ import annotations

import hashlib
import hmac
import re
from pathlib import Path

from .environment_pack import load_environment_pack

COMMITMENT_VERSION = "1.0"
_HEX64 = re.compile(r"[0-9a-f]{64}")


def _check_hex(value: object, what: str) -> str:
    if not isinstance(value, str) or not _HEX64.fullmatch(value):
        raise ValueError(f"{what} must be 64 lowercase hex digits")
    return value


def suite_commitment(pack_digest: str, salt_hex: str) -> str:
    """SHA-256 over the domain tag, the environment pack digest and a 256-bit salt."""
    digest = _check_hex(pack_digest, "pack_digest")
    salt = _check_hex(salt_hex, "salt")
    text = f"emberbench-suite-commitment/{COMMITMENT_VERSION}\n{digest}\n{salt}\n"
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def verify_suite_commitment(pack_dir: str | Path, salt_hex: str, commitment: str) -> bool:
    """True when the verified pack in ``pack_dir`` with ``salt_hex`` opens ``commitment``."""
    pack = load_environment_pack(pack_dir)
    return hmac.compare_digest(suite_commitment(pack.digest, salt_hex), _check_hex(commitment, "commitment"))
