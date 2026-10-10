#!/usr/bin/env bash
set -e
python src/generate_data.py
python src/train_model.py
streamlit run app.py
