# Luxtari AlphaForge — fixture/unit smoke helpers only.
.PHONY: help test
help:
	@echo "Luxtari AlphaForge"
	@echo "  make test   # run pytest (requires local venv)"
test:
	pytest -q
