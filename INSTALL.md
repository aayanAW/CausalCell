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
container runtimes (see the STATE appendix of the manuscript).
