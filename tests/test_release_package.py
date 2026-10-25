"""Release metadata and manifest completeness checks."""
from scripts.check_release import audit


def test_evidence_release_is_complete():
    assert audit() == []
