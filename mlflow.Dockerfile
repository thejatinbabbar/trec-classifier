# Base image
FROM python:3.8-slim

WORKDIR /opt

# Install MLflow and dependencies for model inference
RUN pip install mlflow

# Expose port for serving
EXPOSE 5001

# RUN mkdir -p /mlflow && chmod -R 777 /mlflow