# Course Study Assistant: HTTP API (FastAPI) in a container, following the class sps_genai pattern.
#
#   docker build -t study-assistant .
#   docker run --rm -p 8000:8000 study-assistant          # then open http://localhost:8000/docs
#
# The language model is NOT in the image: by default the API talks to Ollama on the host machine
# (OLLAMA_HOST=http://host.docker.internal:11434), where it can use the Mac's GPU. Or pass
# -e LLM_BACKEND=gemini -e GEMINI_API_KEY=... (see docker-compose.yml and the README).
FROM python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.12.15 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    HF_HOME=/code/.hf

WORKDIR /code

# Dependencies first (cached layer): CPU-only PyTorch on Linux, no dev tools. uv's download cache is a
# build-time mount, so it speeds up rebuilds without ending up in the image.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-install-project

# The package and the knowledge base.
COPY src ./src
COPY data/processed ./data/processed
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev

# Download the pinned embedding model and embed the knowledge base at build time, so the container
# starts quickly and needs no internet access at run time.
RUN .venv/bin/python -c "from study_assistant.retrieval import build_retriever; \
r = build_retriever('hybrid'); print(len(r.chunks), 'chunks;', r.search('warm-up', k=1)[0].chunk.doc_id)"

ENV HF_HUB_OFFLINE=1 \
    LLM_BACKEND=ollama \
    OLLAMA_HOST=http://host.docker.internal:11434 \
    RETRIEVER=hybrid \
    VERIFY=off

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD .venv/bin/python -c \
    "import urllib.request; urllib.request.urlopen('http://localhost:8000/')"
CMD [".venv/bin/uvicorn", "study_assistant.api:app", "--host", "0.0.0.0", "--port", "8000"]
