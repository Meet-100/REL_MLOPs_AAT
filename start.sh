#!/bin/bash

# Ensure MLflow directory exists
mkdir -p mlruns

# Start MLflow UI in the background
echo "Starting MLflow UI on port 5000..."
mlflow ui --host 0.0.0.0 --port 5000 &

# Start FastAPI in the background
echo "Starting FastAPI Inference API on port 8000..."
uvicorn api:app --host 0.0.0.0 --port 8000 &

# Start Streamlit in the foreground
echo "Starting Streamlit Dashboard on port 8501..."
streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0
