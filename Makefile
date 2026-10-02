.PHONY: setup lint format typecheck test test-fast bench run docker-build clean

setup:
	uv sync
	uv run pre-commit install

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

format:
	uv run ruff check --fix src tests
	uv run ruff format src tests

typecheck:
	uv run mypy

test:
	uv run pytest --cov

test-fast:
	uv run pytest -x -q

# Filled in at M1+; kept so the target list matches CONTRIBUTING.md.
bench:
	@echo "bench: no benchmarks yet (TBD, see docs/RESULTS.md)"

run:
	@echo "run: API arrives in M6"

docker-build:
	@echo "docker-build: Dockerfile arrives in M6"

clean:
	rm -rf .mypy_cache .ruff_cache .pytest_cache .hypothesis .coverage htmlcov dist build
