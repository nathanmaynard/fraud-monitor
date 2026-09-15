.PHONY: setup data run test lint app docker
setup:        ## install deps
	uv sync --extra dev
data:         ## download BAF from Kaggle (needs ~/.kaggle/kaggle.json)
	uv run kaggle datasets download -d sgpjesus/bank-account-fraud-dataset-neurips-2022 -p data/raw --unzip
run:          ## full pipeline on real data
	uv run fm all
smoke:        ## full pipeline on synthetic data
	uv run fm all --synthetic
test:
	uv run pytest -q
lint:
	uv run ruff check .
app:
	uv run streamlit run app/streamlit_app.py
docker:
	docker build -t fraud-monitor . && docker run -p 8080:8080 fraud-monitor
