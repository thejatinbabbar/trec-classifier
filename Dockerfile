# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container
COPY Makefile pyproject.toml /app
COPY app /app/app
COPY classifier /app/classifier
COPY experiments/20240821/model.pth /app/model.pth

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip
RUN pip install poetry
RUN poetry install

# Make port 5001 available to the world outside this container
EXPOSE 5001

# Define environment variable
ENV FLASK_APP=app/flask_app.py

# Run the application
CMD ["poetry", "run", "flask", "run", "--host=0.0.0.0", "--port=5001"]
