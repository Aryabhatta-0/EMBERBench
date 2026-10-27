"""Retired public composition API: no official distribution is shipped here.

The previously published construction implementation is exposed historical
research. Its compatibility version labels and opaque digest remain readable
for old public artifacts; they do not identify a future secret policy.
Official construction requires the separately maintained private evaluator.
"""
from typing import NoReturn
from .errors import GenerationError

COMPOSITION_VERSION = "1.0"
SELECTION_RULES_VERSION = "1.0"
RECIPE_CATALOG_VERSION = "1.0"
SUITE_FREEZE_RECORD_VERSION = "1.0"
SUITE_FREEZE_RECORD_NAME = "suite-freeze.json"
_ARCHIVED_PUBLIC_RULE_SET_DIGEST = "61985f4711e28ebd68c7d58140f015e9569ed4eeda819420a1b7e7f8ced0f8e1"


def rule_set_digest() -> str:
    """Opaque identity of the retired, already-public research rules only."""
    return _ARCHIVED_PUBLIC_RULE_SET_DIGEST


def _unavailable(*args: object, **kwargs: object) -> NoReturn:
    raise GenerationError(
        "Official suite construction is evaluator-private. The public package "
        "provides development generation only; no public or archived-policy fallback is allowed."
    )


compose_pool = _unavailable
compose_suite = _unavailable
freeze_composed_suite = _unavailable
load_composed_suite = _unavailable
schedule_composed_worlds = _unavailable
