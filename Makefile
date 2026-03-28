PYTHON  := .venv/bin/python
PIP     := .venv/bin/pip
JUPYTER := .venv/bin/jupyter
STREAMLIT := .venv/bin/streamlit
PYTEST  := .venv/bin/pytest

.PHONY: setup eda preprocess train evaluate dashboard test all clean help

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

setup:  ## Create venv and install dependencies
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "✅ Environment ready. Activate with: source .venv/bin/activate"

test:  ## Run the pytest suite
	$(PYTEST) tests/ -v --tb=short

eda:  ## Launch the EDA notebook
	$(JUPYTER) notebook notebooks/01_eda.ipynb

preprocess:  ## Run data cleaning + feature engineering
	$(PYTHON) -m src.features.pipeline

train:  ## Train all models
	$(PYTHON) -m src.models.baseline
	$(PYTHON) -m src.models.xgboost_model
	$(PYTHON) -m src.models.trainer --quick

evaluate:  ## Run full evaluation suite
	$(PYTHON) -m src.evaluation.metrics
	$(PYTHON) -m src.evaluation.shap_analysis
	$(PYTHON) -m src.evaluation.fairness

dashboard:  ## Launch the Streamlit explainability dashboard
	$(STREAMLIT) run dashboard/app.py

all: preprocess train evaluate  ## Run full pipeline end-to-end

clean:  ## Remove generated artifacts
	rm -rf reports/figures/* reports/metrics/* models/*.joblib
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name '*.pyc' -delete
	@echo "✅ Artifacts cleaned."
