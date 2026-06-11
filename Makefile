.PHONY: setup train evaluate database app clean

setup:
	bash setup_mac.sh

train:
	python scripts/train.py --config configs/default.yaml

evaluate:
	python scripts/evaluate.py --checkpoint checkpoints/best.ckpt --config configs/default.yaml --output_dir outputs

database:
	python scripts/build_database.py --checkpoint checkpoints/best.ckpt

app:
	python frontend/app_gradio.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache
