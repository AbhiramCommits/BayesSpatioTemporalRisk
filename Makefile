PYTHON ?= .venv/bin/python
PIP ?= .venv/bin/pip

.PHONY: all simulate aggregate baselines bayes checks readout report test test-slow lint clean

all: simulate aggregate baselines bayes checks readout
	@echo "Pipeline complete: reports/risk_readout.md"

simulate:
	$(PYTHON) scripts/01_simulate.py

aggregate: simulate
	$(PYTHON) scripts/02_aggregate.py

baselines: aggregate
	$(PYTHON) scripts/03_baselines.py

bayes: baselines
	$(PYTHON) scripts/04_fit_bayes.py

checks: bayes
	$(PYTHON) scripts/05_checks.py

readout: checks
	$(PYTHON) scripts/06_readout.py

report:
	$(PYTHON) scripts/06_readout.py
	@echo "Wrote reports/risk_readout.md"

test:
	$(PYTHON) -m pytest -m "not slow" -q

test-slow:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check .

clean:
	rm -f data/*.parquet data/*.nc
	rm -f artifacts/*.nc artifacts/*.json artifacts/*.csv artifacts/*.parquet
	rm -f figures/*.png reports/*.md
