FROM python:3.9-slim

# Set the working directory
WORKDIR /app

RUN pip install --upgrade pip
RUN pip install boto3