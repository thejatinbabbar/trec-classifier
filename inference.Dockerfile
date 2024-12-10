FROM python:3.9-slim

WORKDIR /app

COPY pyproject.toml /app/
COPY app /app/app
COPY classifier /app/classifier
COPY config /app/config
COPY artifacts /app/artifacts

RUN pip install --upgrade pip
RUN pip install poetry
RUN poetry install

ENV FLASK_APP=app/flask_app.py

EXPOSE 5000

CMD ["poetry", "run", "flask", "run"]