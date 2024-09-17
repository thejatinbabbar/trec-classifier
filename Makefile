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

run_training:
	docker compose up --build training

run_app:
	docker compose up --build inference