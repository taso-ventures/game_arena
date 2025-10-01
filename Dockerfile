# Game Arena Docker Container
# Supports Chess, Go, and FreeCiv gameplay with LLM agents

FROM python:3.11-slim

# Install system dependencies including CMake and Clang for OpenSpiel
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    clang \
    curl \
    git \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY pyproject.toml ./

# Install additional dependencies for FreeCiv integration
RUN pip install --no-cache-dir \
    websockets \
    aiohttp \
    requests

# Copy the application code
COPY . .

# Install in development mode with all dev dependencies
RUN pip install --no-cache-dir -e .[dev]

# Ensure critical dependencies are available (redundant safety check)
RUN pip install --no-cache-dir termcolor absl-py

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default command (can be overridden)
CMD ["python", "-m", "game_arena.harness.harness_demo", "--help"]