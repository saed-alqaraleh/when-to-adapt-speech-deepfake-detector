.PHONY: verify integrity test release-check
verify:
	python scripts/verify_all.py
integrity:
	python scripts/verify_integrity.py
test:
	python -m pytest -q
release-check:
	python scripts/check_release.py
