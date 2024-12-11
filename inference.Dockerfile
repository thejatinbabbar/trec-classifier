FROM python:3.9-slim

WORKDIR /app

COPY pyproject.toml /app/
COPY app /app/app
COPY classifier /app/classifier
COPY config /app/config
COPY artifacts /app/artifacts

RUN pip install --upgrade pip --root-user-action=ignore
RUN pip install poetry --root-user-action=ignore
RUN poetry install

ENV FLASK_APP=app/flask_app.py

EXPOSE 5000

CMD ["poetry", "run", "flask", "run", "--host=0.0.0.0", "--port=5000"]