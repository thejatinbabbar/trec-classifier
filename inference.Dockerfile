FROM python:3.9-slim

WORKDIR /app

COPY app/pyproject.toml /app/
COPY app/poetry.lock /app/

RUN pip install --upgrade pip --root-user-action=ignore
RUN pip install poetry --root-user-action=ignore
RUN poetry install

COPY artifacts /app/artifacts
COPY classifier /app/classifier
COPY app /app/app
COPY config /app/config
COPY data /app/data

EXPOSE 8501

ENV PYTHONPATH=${PYTHONPATH}:/app

CMD ["poetry", "run", "streamlit", "run", "/app/app/streamlit_app.py"]