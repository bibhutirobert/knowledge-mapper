.PHONY: install model run test lint clean zip

# ── Setup ──────────────────────────────────────────────────────────────────
install:
	pip install -r requirements.txt

model:
	python -m spacy download en_core_web_sm

setup: install model
	@echo "✓ Environment ready. Run: make run"

# ── App ────────────────────────────────────────────────────────────────────
run:
	streamlit run app.py

run-debug:
	PYTHONPATH=. streamlit run app.py --logger.level debug

# ── CLI pipeline ──────────────────────────────────────────────────────────
cli:
	@echo "Usage: python pipeline.py <input.pdf> [--output ./output] [--formats json html md pdf]"

# ── Tests ──────────────────────────────────────────────────────────────────
test:
	python -m pytest tests/ -v

test-fast:
	python -m pytest tests/ -v --tb=short -x

test-coverage:
	python -m pytest tests/ --cov=. --cov-report=term-missing --cov-report=html

# ── Code quality ──────────────────────────────────────────────────────────
lint:
	python -m flake8 . --max-line-length=110 --exclude=.venv,__pycache__

format:
	python -m black . --line-length=110

# ── Packaging ─────────────────────────────────────────────────────────────
zip:
	zip -r knowledge_mapper.zip . \
	  --exclude ".venv/*" --exclude "__pycache__/*" \
	  --exclude "*.pyc" --exclude ".git/*" \
	  --exclude "logs/*" --exclude "output/*" \
	  --exclude "*.egg-info/*"
	@echo "✓ knowledge_mapper.zip created"

# ── Clean ─────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf logs/ output/ .pytest_cache/ htmlcov/ .coverage
	@echo "✓ Cleaned"
