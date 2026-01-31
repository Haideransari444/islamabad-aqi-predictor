"""
Main training script for AQI prediction models.
"""
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.features.feature_store import get_feature_store
from src.features.compute_features import compute_all_features
from src.training.models.sklearn_models import get_all_models, get_model
from src.training.evaluate import evaluate_model, compare_models, print_evaluation_report, get_best_model
from src.training.model_registry import get_model_registry


def load_training_data(
    feature_group_name: str = 'aqi_features',
    start_date: datetime = None,
    end_date: datetime = None
) -> pd.DataFrame:
    """
    Load training data from feature store.
    
    Args:
        feature_group_name: Name of the feature group
        start_date: Start date for data
        end_date: End date for data
        
    Returns:
        DataFrame with features and targets
    """
    fs = get_feature_store()
    df = fs.get_features(feature_group_name, start_date, end_date)
    
    if df.empty:
        raise ValueError("No training data found in feature store")
    
    return df


def prepare_data(
    df: pd.DataFrame,
    target_col: str = 'aqi_target_24h',
    test_size: float = 0.2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    """
    Prepare data for training.
    
    Args:
        df: DataFrame with features and targets
        target_col: Target column name
        test_size: Fraction of data for testing
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, feature_names)
    """
    # Process features if not already done
    if 'hour' not in df.columns:
        df, feature_cols, target_cols = compute_all_features(df)
    else:
        exclude_cols = ['timestamp', 'city']
        target_cols = [col for col in df.columns if '_target_' in col]
        feature_cols = [col for col in df.columns if col not in exclude_cols + target_cols]
    
    # Remove rows with NaN values
    df = df.dropna()
    
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found. Available: {target_cols}")
    
    X = df[feature_cols].values
    y = df[target_col].values
    
    # Time series split (don't shuffle)
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    return X_train, X_test, y_train, y_test, feature_cols


def train_single_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    params: Dict[str, Any] = None
) -> Tuple[Any, Dict[str, float]]:
    """
    Train a single model.
    
    Args:
        model_name: Name of the model
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        y_test: Test targets
        params: Model hyperparameters
        
    Returns:
        Tuple of (trained model, metrics)
    """
    print(f"\nTraining {model_name}...")
    
    model = get_model(model_name, params)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    metrics = evaluate_model(y_test, y_pred)
    
    print_evaluation_report(metrics, model_name)
    
    return model, metrics


def train_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, Tuple[Any, Dict[str, float]]]:
    """
    Train all available models.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        y_test: Test targets
        
    Returns:
        Dictionary of model name to (model, metrics)
    """
    results = {}
    models = get_all_models()
    
    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        metrics = evaluate_model(y_test, y_pred)
        
        print_evaluation_report(metrics, model_name)
        results[model_name] = (model, metrics)
    
    return results


def save_best_model(
    results: Dict[str, Tuple[Any, Dict[str, float]]],
    metric: str = 'rmse'
) -> str:
    """
    Save the best performing model.
    
    Args:
        results: Dictionary of model results
        metric: Metric to use for comparison
        
    Returns:
        Path to saved model
    """
    # Extract just metrics for comparison
    metrics_only = {name: m for name, (_, m) in results.items()}
    best_model_name = get_best_model(metrics_only, metric)
    
    best_model, best_metrics = results[best_model_name]
    
    print(f"\nBest model: {best_model_name} (based on {metric})")
    
    # Save to registry
    registry = get_model_registry()
    model_path = registry.save_model(
        model=best_model,
        model_name=best_model_name,
        metrics=best_metrics
    )
    
    return model_path


def main():
    """Main training pipeline."""
    print("="*60)
    print("AQI Prediction Model Training Pipeline")
    print("="*60)
    
    # Configuration
    target_col = 'aqi_target_24h'  # Predict 24 hours ahead
    
    # Load data
    print("\n1. Loading training data...")
    try:
        df = load_training_data()
        print(f"   Loaded {len(df)} records")
    except Exception as e:
        print(f"   Error loading data: {e}")
        print("   Using sample data for demonstration...")
        
        # Create sample data
        dates = pd.date_range(start='2024-01-01', periods=1000, freq='H')
        df = pd.DataFrame({
            'timestamp': dates,
            'city': 'Karachi',
            'aqi': np.random.randint(50, 200, 1000),
            'pm2_5': np.random.uniform(10, 100, 1000),
            'pm10': np.random.uniform(20, 150, 1000),
        })
    
    # Prepare data
    print("\n2. Preparing features and targets...")
    X_train, X_test, y_train, y_test, feature_names = prepare_data(df, target_col)
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    print(f"   Number of features: {len(feature_names)}")
    
    # Train all models
    print("\n3. Training models...")
    results = train_all_models(X_train, y_train, X_test, y_test)
    
    # Save best model
    print("\n4. Saving best model...")
    model_path = save_best_model(results)
    
    print("\n" + "="*60)
    print("Training pipeline completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
