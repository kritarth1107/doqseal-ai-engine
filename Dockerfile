# DoqSeal AI engine — API + worker (models/GPU hosting comes later)
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=3031 \
    HOST=0.0.0.0

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# App deps first. sentence-transformers pulls a PyPI torchvision that does not
# match a CPU torch, and import then dies with
# "operator torchvision::nms does not exist" (document search never embeds).
# Force a matched CPU pair last so nothing can replace it.
RUN pip install --upgrade pip \
    && grep -vE '^(torch|torchvision)([=<>!]|$)' requirements.txt > /tmp/requirements.notorch.txt \
    && pip install -r /tmp/requirements.notorch.txt \
    && pip install --force-reinstall --no-deps \
        "torch==2.6.0" "torchvision==0.21.0" \
        --index-url https://download.pytorch.org/whl/cpu \
    && rm /tmp/requirements.notorch.txt \
    && python -c "from torchvision.ops import nms; import sentence_transformers; print('torchvision nms ok')"

COPY app ./app
COPY scripts/docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

EXPOSE 3031
ENTRYPOINT ["/docker-entrypoint.sh"]
