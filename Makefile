.PHONY: install lint format format-check test coverage docs-check ci

install:
    uv sync --dev

lint:
    uv run ruff check .

format:
    uv run ruff format .

format-check:
    uv run ruff format --check .

test:
    uv run pytest

coverage:
    uv run pytest --cov=src --cov-report=term-missing

docs-check:
    test -f README.md
    test -f CONTRIBUTING.md
    test -f CHANGELOG.md

ci: lint format-check test docs-check
