# syntax=docker/dockerfile:1.4
FROM python:3.11-slim AS builder

# Install necessary system dependencies
RUN apt-get update && apt-get install -y \
    git \
    gcc \
    g++ \
    musl-dev \
    graphviz \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Add docker group and vscode user
RUN groupadd -r docker && useradd -r -g docker -s /bin/bash vscode

# Copy .env file to /src folder
COPY .env /src/.env

WORKDIR /src

# Copy requirements.txt and install Python dependencies
COPY ./src/requirements.txt /src
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Copy application code
COPY ./src/hyperopt_generate_pipeline.py /src
COPY ./src/tpot_generate_pipeline.py /src
COPY ../example /src/example
COPY ./src/. .

# Install additional Python dependencies
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir numpy scikit-learn

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install git+https://github.com/hyperopt/hyperopt-sklearn

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install svgwrite

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir  tpot

# Set environment variable for Flask server port
ENV FLASK_SERVER_PORT=9000

# Expose the Flask server port
EXPOSE 9000

# Command to run the Flask application
CMD ["python3", "server.py"]
