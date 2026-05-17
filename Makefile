.PHONY: help install test run-web run-api run-notebook clean train demo docker-build docker-run

help:
	@echo "🚀 SpaceX Falcon 9 Landing Prediction - Make Commands"
	@echo ""
	@echo "  make install        - Install dependencies"
	@echo "  make test           - Run all tests"
	@echo "  make train          - Train the model"
	@echo "  make demo           - Run demo script"
	@echo "  make run-web        - Run Streamlit web app"
	@echo "  make run-api        - Run Flask API"
	@echo "  make run-notebook   - Run Jupyter notebooks"
	@echo "  make clean          - Clean cache files"
	@echo "  make docker-build   - Build Docker image"
	@echo "  make docker-run     - Run Docker container"

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v

train:
	python src/main.py

demo:
	python demo.py

run-web:
	streamlit run deployment/streamlit/app.py

run-api:
	python deployment/flask/app.py

run-notebook:
	jupyter lab notebooks/

run-dashboard:
	streamlit run dashboard/dashboard.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	@echo "✅ Cleaned"

docker-build:
	docker build -t spacex-prediction:latest .

docker-run:
	docker run -p 8501:8501 spacex-prediction:latest

lint:
	flake8 src/ --count --select=E9,F63,F7,F82 --show-source --statistics

format:
	black src/ tests/

cov:
	pytest tests/ --cov=src --cov-report=html --cov-report=term

install-dev:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pre-commit install

setup:
	python -m venv venv
	./venv/bin/pip install -r requirements.txt
	@echo "Run 'source venv/bin/activate' to activate the environment"