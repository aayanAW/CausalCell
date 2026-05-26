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
