# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# # Copy the current directory contents into the container
COPY Makefile pyproject.toml /app/

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip
RUN pip install poetry
RUN poetry install

# EXPOSE 5001

# # Run the application
# CMD ["poetry", "run", "python", "classifier/main.py", "--experiment_name", "trec-classification", "--config", "config/config.yml"]
