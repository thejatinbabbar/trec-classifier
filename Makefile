# Makefile for formatting and linting Python code

install:
	pip install --upgrade pip && \
	pip install poetry && \
	poetry install

# Formatting with Black
black:
	poetry run black classifier/

# Sorting imports with isort
isort:
	poetry run isort classifier/

# Linting with flake8
flake8:
	poetry run flake8 classifier/

# Run all formatting and linting
format: black isort

# Check formatting and linting
check:
	poetry run black --check classifier/ ; \
	poetry run isort --check-only classifier/ ; \
	poetry run flake8 classifier/

test:
	poetry run pytest --cov=classifier --cov=app

build_flask_app:
	docker build -t trec-classifier-flask .

run_flask_app:
	docker run -p 5001:5001 trec-classifier-flask