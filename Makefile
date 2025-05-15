train:
	uv run python3 -m scripts.run_training

translate:
	uv run python3 -m scripts.run_translation

preprocess:
	uv run python3 -m scripts.run_preprocess

test:
	uv run pytest tests/

clean:
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete

clear_logs:
	rm -rf ./logs/*

log:
	PYTHONPATH=. uv run tensorboard --logdir logs/
