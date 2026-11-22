FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 ARXIV_RAG_DATA_DIR=/data ARXIV_RAG_RESULTS_DIR=/data/results

COPY pyproject.toml README.md ./
COPY src ./src

# Build with dense retrieval: docker build --build-arg EXTRAS=dense -t arxiv-rag .
ARG EXTRAS=""
RUN pip install --no-cache-dir ".${EXTRAS:+[$EXTRAS]}"

VOLUME /data
ENTRYPOINT ["arxiv-rag"]
CMD ["--help"]
