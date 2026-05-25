#!/usr/bin/env bash
# Phase 8: Engineering — pyproject.toml, Makefile, Dockerfile, INSTALL.md,
# fork→spawn, modal_runner consolidation.
set -u
ROOT="/Users/aayanalwani/virtual cells"
cd "$ROOT" || exit 1

# pyproject.toml
if [ ! -f "$ROOT/pyproject.toml" ]; then
cat > "$ROOT/pyproject.toml" <<'TOML'
[build-system]
requires = ["setuptools>=61", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "perturbcausal"
version = "1.0.0"
description = "Benchmark for causal-recovery substitutability of virtual cell model predictions"
requires-python = ">=3.9,<3.13"
license = {text = "MIT"}
authors = [{name = "Anonymous"}]
dependencies = [
  "numpy==1.26.4",
  "scipy==1.13.1",
  "scikit-learn>=1.5,<1.7",
  "anndata==0.10.9",
  "scanpy==1.10.3",
  "gies==0.0.1",
  "networkx>=3.0,<4.0",
  "causal-learn>=0.1,<0.2",
  "arboreto==0.1.6",
]

[project.optional-dependencies]
state = [
  "transformers==4.52.3",
  "tokenizers>=0.21,<0.22",
  "huggingface_hub>=0.30,<0.40",
  "torch>=2.4,<2.7",
]
dev = ["pytest>=8", "ruff>=0.5"]
TOML
fi

# requirements-modal.txt
if [ ! -f "$ROOT/requirements-modal.txt" ]; then
cat > "$ROOT/requirements-modal.txt" <<'TXT'
numpy==1.26.4
scipy==1.13.1
scikit-learn>=1.5,<1.7
anndata==0.10.9
scanpy==1.10.3
gies==0.0.1
networkx>=3.0,<4.0
causal-learn>=0.1,<0.2
arboreto==0.1.6
transformers==4.52.3
torch==2.4.1
TXT
fi

# Makefile
if [ ! -f "$ROOT/Makefile" ]; then
cat > "$ROOT/Makefile" <<'MK'
.PHONY: env test repro figures clean

env:
	python3 -m venv .venv_v4
	./.venv_v4/bin/pip install -e .

test:
	./.venv_v4/bin/pytest tests/ -v

repro: env
	./.venv_v4/bin/python scripts/run_eval_local.py \
		--data-path data/k562_subset_n200_slim.h5ad \
		--pre-subset --n-genes 200 --partition-size 20 \
		--n-cells 100 --n-ctrl 200 --seed 42 \
		--model-outputs-dir data/model_outputs_real_n200 \
		--gies-cache-dir results/gies_cache --skip-random
	./.venv_v4/bin/python scripts/aggregate_multi_seed.py

figures:
	./.venv_v4/bin/python scripts/generate_figures.py

clean:
	rm -rf .venv_v4 build dist *.egg-info __pycache__ */__pycache__
MK
fi

# Dockerfile
if [ ! -f "$ROOT/Dockerfile" ]; then
cat > "$ROOT/Dockerfile" <<'DOCKER'
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
DOCKER
fi

# INSTALL.md
if [ ! -f "$ROOT/INSTALL.md" ]; then
cat > "$ROOT/INSTALL.md" <<'MD'
# PerturbCausal Installation

## Quick start (Docker — recommended)

```
docker build -t perturbcausal .
docker run --rm -v $(pwd)/results:/app/results perturbcausal
```

## Native install

```
python3 -m venv .venv_v4
./.venv_v4/bin/pip install -e .
./.venv_v4/bin/pip install -e .[state]  # for STATE inference
```

R + inspre (optional, for the inspre backend):

```
brew install r  # or apt install r-base
R -e 'install.packages("inspre")'
```

### STATE compatibility patch

After installing transformers==4.52.3, apply the in-source patch:

```
python3 -c "import transformers, os, re; \
  p = os.path.join(os.path.dirname(transformers.__file__), 'modeling_utils.py'); \
  s = open(p).read(); \
  open(p, 'w').write(re.sub(r'if v not in ALL_PARALLEL_STYLES:', \
    'if ALL_PARALLEL_STYLES is not None and v not in ALL_PARALLEL_STYLES:', s))"
```

This works around a known incompatibility between transformers 4.52.3 and Modal
container runtimes (see ARCHITECTURE_V4_AUDIT_AND_UPGRADE.md and the STATE
appendix of the manuscript).
MD
fi

# fork→spawn in run_eval_local.py
if grep -q 'mp.get_context("fork")' "$ROOT/scripts/run_eval_local.py" 2>/dev/null; then
    sed -i.bak 's/mp.get_context("fork")/mp.get_context("spawn")/' "$ROOT/scripts/run_eval_local.py"
    echo "  fork → spawn applied"
fi

echo "  Phase 8: env files + Docker + Makefile written"
