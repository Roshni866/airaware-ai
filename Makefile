.PHONY: test data features train evaluate app reproduce clean

test:
	python -m pytest -q

data:
	python -m src.data_pipeline

features:
	python -m src.features

train:
	python -m src.train

evaluate:
	python -m src.evaluate

app:
	streamlit run app.py

reproduce: test data features train evaluate
	@echo "Done. See REPORT.md and results/."

clean:
	rm -rf data/processed/* results/*.csv results/*.json results/figures/*.png models/*.joblib
