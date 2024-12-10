# Makefile for formatting and linting Python code

clean:
	find . -type d -empty -delete
	find . -name "*.pyc" -delete
	find . -name "*.log" -delete
	find . -name "*.tmp" -delete
	find . -name "*.swp" -delete
	find . -name "__pycache__" -type d -exec rm -r {} +

install:
	pip install --upgrade pip && \
	pip install poetry && \
	poetry install

# Formatting with Black
black:
	poetry run black classifier/ config/ scripts/ app/ tests/

# Sorting imports with isort
isort:
	poetry run isort classifier/ config/ scripts/ app/ tests/

# Linting with flake8
flake8:
	poetry run flake8 classifier/ config/ scripts/ app/ tests/

# Run all formatting and linting
format: black isort

# Check formatting and linting
check:
	poetry run black --check classifier/ config/ scripts/ app/ tests/ ; \
	poetry run isort --check-only classifier/ config/ scripts/ app/ tests/ ; \
	poetry run flake8 classifier/ config/ scripts/ app/ tests/

test:
	poetry run pytest --cov=classifier --cov=app

build_app:
	docker build -t trec-inference . -f inference.Dockerfile

run_training:
	docker compose up --build training

run_app:
	docker run -p 5000:5000 trec-inference