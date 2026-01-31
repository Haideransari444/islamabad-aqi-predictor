# Pearls AQI Predictor

Predict the Air Quality Index (AQI) in your city for the next 3 days using a 100% serverless stack.

## Project Structure

```
├── data/                       # Data storage
│   ├── raw/                    # Raw AQI data from APIs
│   ├── processed/              # Processed features and targets
│   └── backfill/               # Historical backfilled data
│
├── src/                        # Source code
│   ├── features/               # Feature pipeline
│   │   ├── __init__.py
│   │   ├── fetch_data.py       # Fetch raw AQI data from APIs
│   │   ├── compute_features.py # Compute model inputs (features)
│   │   └── feature_store.py    # Store features (Hopsworks/Vertex AI)
│   │
│   ├── training/               # Training pipeline
│   │   ├── __init__.py
│   │   ├── train.py            # Main training script
│   │   ├── models/             # Model definitions
│   │   │   ├── __init__.py
│   │   │   ├── sklearn_models.py   # Random Forest, Ridge Regression
│   │   │   └── deep_learning.py    # TensorFlow/PyTorch models
│   │   ├── evaluate.py         # RMSE, MAE, R² evaluation
│   │   └── model_registry.py   # Save/load models
│   │
│   ├── inference/              # Inference pipeline
│   │   ├── __init__.py
│   │   ├── predict.py          # Make predictions
│   │   └── explainability.py   # SHAP/LIME explanations
│   │
│   └── utils/                  # Utility functions
│       ├── __init__.py
│       ├── config.py           # Configuration settings
│       └── logger.py           # Logging utilities
│
├── notebooks/                  # Jupyter notebooks
│   ├── 01_eda.ipynb           # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_experiments.ipynb
│
├── webapp/                     # Web application
│   ├── app.py                  # Main Streamlit/Gradio app
│   ├── api/                    # FastAPI/Flask backend
│   │   ├── __init__.py
│   │   └── main.py
│   ├── static/                 # Static assets
│   └── templates/              # HTML templates (if using Flask)
│
├── pipelines/                  # CI/CD and automation
│   ├── feature_pipeline.py     # Scheduled feature extraction
│   ├── training_pipeline.py    # Scheduled model training
│   └── inference_pipeline.py   # Scheduled predictions
│
├── .github/                    # GitHub Actions
│   └── workflows/
│       ├── feature_pipeline.yml    # Hourly feature extraction
│       └── training_pipeline.yml   # Daily model training
│
├── tests/                      # Unit and integration tests
│   ├── __init__.py
│   ├── test_features.py
│   ├── test_training.py
│   └── test_inference.py
│
├── configs/                    # Configuration files
│   ├── config.yaml             # Main configuration
│   └── model_config.yaml       # Model hyperparameters
│
├── requirements.txt            # Python dependencies
├── setup.py                    # Package setup
├── .env.example                # Environment variables template
├── .gitignore                  # Git ignore file
└── README.md                   # This file
```

## Features

- **Feature Pipeline**: Fetches AQI data, computes time-based features (hour, day, month), AQI change rate
- **Training Pipeline**: Supports Scikit-learn (Random Forest, Ridge Regression) and Deep Learning models
- **Automated Pipelines**: GitHub Actions for hourly feature updates and daily model retraining
- **Web Dashboard**: Interactive UI showing predictions and forecasts
- **Explainability**: SHAP/LIME for feature importance
- **Alerts**: Notifications for hazardous AQI levels

## Guidelines

1. Perform EDA to identify trends
2. Use variety of forecasting models (statistical to deep learning)
3. Use SHAP or LIME for feature importance explanations
4. Add alerts for hazardous AQI levels

## Final Deliverables

1. End-to-end AQI prediction system
2. A scalable, automated pipeline
3. An interactive dashboard showcasing real-time and forecasted AQI data

## Setup

1. Clone this repository
2. Create virtual environment: `python -m venv venv`
3. Activate: `venv\Scripts\activate` (Windows)
4. Install dependencies: `pip install -r requirements.txt`
5. Copy `.env.example` to `.env` and configure
6. Run the app: `streamlit run webapp/app.py`
