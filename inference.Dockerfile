FROM python:3.9-slim

WORKDIR /app

COPY app/pyproject.toml /app/
COPY app/poetry.lock /app/

RUN pip install --upgrade pip --root-user-action=ignore
RUN pip install poetry --root-user-action=ignore
RUN poetry config virtualenvs.create false && poetry install --no-dev --no-interaction --no-ansi

COPY artifacts /app/artifacts
COPY pipeline /app/pipeline
COPY config /app/config
COPY data /app/data
COPY app /app/app

EXPOSE 8501

ENV PYTHONPATH=/app

CMD ["poetry", "run", "streamlit", "run", "/app/app/streamlit_app.py"]