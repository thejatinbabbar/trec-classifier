# Use a minimal base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy only the necessary files
# COPY pyproject.toml /app/
# COPY app /app/app
# COPY classifier /app/classifier
# COPY config /app/config

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip
RUN pip install poetry
RUN poetry install

# Set environment variables
ENV FLASK_ENV=production
ENV FLASK_APP=app/flask_app.py

# Expose port
EXPOSE 5000

# Run the application
# CMD ["poetry", "run", "flask", "run"]
