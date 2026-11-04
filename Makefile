# NanoEdge-AD Makefile
PYTHON = .venv/Scripts/python.exe

.PHONY: setup data train quantize prune simulate benchmark evaluate ablation report test all clean

setup:
	$(PYTHON) -m pip install -r requirements.txt

data:
	$(PYTHON) -m src.data_gen

train:
	$(PYTHON) -m src.train

quantize:
	$(PYTHON) -m src.quantize

prune:
	$(PYTHON) -m src.prune_experiment

simulate:
	$(PYTHON) -m src.runtime_pruning

benchmark:
	$(PYTHON) -m src.benchmark

evaluate:
	$(PYTHON) -m src.evaluate

ablation:
	$(PYTHON) -m src.ablation

report:
	$(PYTHON) -m src.report

test:
	$(PYTHON) -m pytest tests/ -v

all: data train quantize prune simulate benchmark evaluate ablation report test

clean:
	rm -rf __pycache__ src/__pycache__ tests/__pycache__ .pytest_cache
