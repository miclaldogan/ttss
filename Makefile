PYTHON ?= python
PIP ?= pip
CONFIG ?= configs/base.yaml
VIDEO ?= sample.mp4

.PHONY: install test lint train demo clean

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install flake8 pytest-cov

test:
	$(PYTHON) -m pytest --cov=ttss --cov-report=term-missing --cov-report=xml -q

lint:
	flake8 ttss scripts tests

train:
	@DATA_ROOT=$$($(PYTHON) -c 'import yaml; print(yaml.safe_load(open("$(CONFIG)", encoding="utf-8"))["data"]["processed_dir"])'); \
	ANNOTATIONS=$$($(PYTHON) -c 'import yaml; print(yaml.safe_load(open("$(CONFIG)", encoding="utf-8"))["data"]["prepared_annotations_csv"])'); \
	EPOCHS=$$($(PYTHON) -c 'import yaml; print(yaml.safe_load(open("$(CONFIG)", encoding="utf-8"))["training"]["epochs"])'); \
	BATCH_SIZE=$$($(PYTHON) -c 'import yaml; print(yaml.safe_load(open("$(CONFIG)", encoding="utf-8"))["training"]["batch_size"])'); \
	LEARNING_RATE=$$($(PYTHON) -c 'import yaml; print(yaml.safe_load(open("$(CONFIG)", encoding="utf-8"))["training"]["learning_rate"])'); \
	echo "$(PYTHON) scripts/train.py --data-root $$DATA_ROOT --annotations $$ANNOTATIONS --epochs $$EPOCHS --batch-size $$BATCH_SIZE --learning-rate $$LEARNING_RATE"; \
	$(PYTHON) scripts/train.py --data-root "$$DATA_ROOT" --annotations "$$ANNOTATIONS" --epochs "$$EPOCHS" --batch-size "$$BATCH_SIZE" --learning-rate "$$LEARNING_RATE"

demo:
	$(PYTHON) scripts/infer.py --video $(VIDEO) --crime-start 0 --crime-end 1 --frame-index 0

clean:
	rm -rf .pytest_cache .coverage coverage.xml htmlcov
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete