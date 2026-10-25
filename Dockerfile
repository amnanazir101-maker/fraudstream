FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FRAUDSTREAM_DATA_DIR=/data

# LightGBM needs the OpenMP runtime.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install ".[dashboard,kafka]"
COPY dashboard.py ./

EXPOSE 8000 8501 5000
CMD ["fraudstream", "serve", "--host", "0.0.0.0"]
