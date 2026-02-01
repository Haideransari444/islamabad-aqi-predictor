"""
Training Pipeline - Scheduled script for daily model training.
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta

sys.path.append(str(Path(__file__).parent.parent))

from src.training.train import (
    load_training_data, 
    prepare_data, 
    train_all_models, 
    save_best_model
)
from src.utils.logger import get_logger

logger = get_logger("training_pipeline")


def run_training_pipeline(
    target_col: str = 'target_24h',
    min_samples: int = 100
):
    """
    Run the training pipeline.
    
    Args:
        target_col: Target column for prediction
        min_samples: Minimum samples required for training
    """
    logger.info("="*50)
    logger.info("Starting Training Pipeline")
    logger.info(f"Target: {target_col}")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info("="*50)
    
    try:
        # Load data
        logger.info("Loading training data...")
        df = load_training_data()
        logger.info(f"  ✓ Loaded {len(df)} records")
        
        if len(df) < min_samples:
            logger.warning(f"Insufficient data ({len(df)} < {min_samples}). Skipping training.")
            return
        
        # Prepare data
        logger.info("Preparing features and targets...")
        X_train, X_test, y_train, y_test, feature_names = prepare_data(df, target_col)
        logger.info(f"  ✓ Training samples: {len(X_train)}")
        logger.info(f"  ✓ Test samples: {len(X_test)}")
        logger.info(f"  ✓ Features: {len(feature_names)}")
        
        # Train models
        logger.info("Training models...")
        results = train_all_models(X_train, y_train, X_test, y_test)
        
        for model_name, (_, metrics) in results.items():
            logger.info(f"  {model_name}: RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}")
        
        # Save best model
        logger.info("Saving best model...")
        model_path = save_best_model(results)
        logger.info(f"  ✓ Model saved to: {model_path}")
        
        logger.info("Training pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        raise
    
    logger.info("="*50)


if __name__ == "__main__":
    run_training_pipeline()
