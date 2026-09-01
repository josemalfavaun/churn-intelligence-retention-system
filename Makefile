.PHONY: install test run-app clean-data features train predict

install:
	pip install -r requirements.txt

features:
	python -m src.features.build_features

train:
	python -m src.models.train_model

predict:
	python -m src.models.predict

test:
	pytest tests/

run-app:
	streamlit run app/streamlit_app.py