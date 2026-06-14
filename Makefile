.PHONY: install test run-app

install:
	pip install -r requirements.txt

test:
	pytest tests/

run-app:
	streamlit run app/streamlit_app.py