FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl r-base \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml requirements-modal.txt ./
RUN pip install --no-cache-dir -r requirements-modal.txt

# Apply transformers source-patch for STATE compatibility
RUN python3 -c "import transformers, os, re; \
    p = os.path.join(os.path.dirname(transformers.__file__), 'modeling_utils.py'); \
    s = open(p).read(); \
    s2 = re.sub(r'if v not in ALL_PARALLEL_STYLES:', \
                'if ALL_PARALLEL_STYLES is not None and v not in ALL_PARALLEL_STYLES:', s); \
    open(p, 'w').write(s2)"

COPY . /app
RUN pip install -e .

CMD ["bash", "-lc", "make repro"]
