"""
Functions for training the curator logistic regression model.
"""

import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    log_loss,
    roc_auc_score,
)


logger = logging.getLogger(__name__)


def _expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Calculate Expected Calibration Error (ECE).

    ECE measures how well predicted probabilities match actual frequencies by binning
    predictions and comparing average confidence to average accuracy in each bin.

    Args:
        y_true: True binary labels (0 or 1).
        y_prob: Predicted probabilities for the positive class.
        n_bins: Number of bins to use for calibration curve.

    Returns:
        ECE value (lower is better, 0 = perfect calibration).
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        # Last bin includes upper boundary
        mask = (y_prob >= lo) & (y_prob < hi if i < n_bins - 1 else y_prob <= hi)

        if np.any(mask):
            acc = y_true[mask].mean()  # Actual accuracy in bin
            conf = y_prob[mask].mean()  # Average confidence in bin
            weight = mask.mean()  # Proportion of samples in bin
            ece += np.abs(acc - conf) * weight

    return float(ece)


def _load_and_prepare_data(training_data_file: str) -> tuple:
    """
    Load training data and prepare feature matrices.

    Args:
        training_data_file: Path to joblib file containing training data.

    Returns:
        Tuple of (X_train, y_train, X_val, y_val).
    """
    logger.info(f"Loading training data from {training_data_file}")
    training_data = joblib.load(training_data_file)

    train_data = training_data["train"]
    val_data = training_data["val"]

    # Concatenate context and movie features
    X_train = np.concatenate(
        [train_data["context_features"], train_data["movie_features"]], axis=1
    )
    y_train = train_data["curator_targets"].astype(int).ravel()

    X_val = np.concatenate(
        [val_data["context_features"], val_data["movie_features"]], axis=1
    )
    y_val = val_data["curator_targets"].astype(int).ravel()

    logger.info(f"Training data shape: {X_train.shape}")
    logger.info(
        f"Training class distribution - Positive: {np.sum(y_train)}, "
        f"Negative: {len(y_train) - np.sum(y_train)}"
    )
    logger.info(f"Validation data shape: {X_val.shape}")

    return X_train, y_train, X_val, y_val


def _train_calibrated_model(
    X_train: np.ndarray, y_train: np.ndarray, cv: int = 5
) -> CalibratedClassifierCV:
    """
    Train a calibrated logistic regression model using cross-validation.

    Uses sigmoid (Platt scaling) calibration with CV, which:
    - Preserves ROC-AUC (discrimination ability)
    - Dramatically improves ECE (calibration quality)
    - Avoids the data-hungry nature of isotonic calibration
    - Uses all training data efficiently via cross-validation

    Args:
        X_train: Training features.
        y_train: Training labels.
        cv: Number of cross-validation folds for calibration.

    Returns:
        Calibrated CalibratedClassifierCV model.
    """
    logger.info(f"Training calibrated model with {cv}-fold CV sigmoid calibration...")

    base_model = LogisticRegression(
        class_weight="balanced",
        max_iter=5000,
        random_state=42,
    )

    calibrated_model = CalibratedClassifierCV(
        estimator=base_model,
        method="sigmoid",  # Platt scaling - works well with small datasets
        cv=cv,
    )

    calibrated_model.fit(X_train, y_train)
    return calibrated_model


def load_curator_model(model_file: str) -> CalibratedClassifierCV:
    """
    Load a trained and calibrated curator logistic regression model.

    Args:
        model_file: Path to the saved model joblib file.

    Returns:
        Loaded CalibratedClassifierCV model ready for prediction.

    Example:
        >>> model = load_curator_model("data/models/curator_logistic_model.joblib")
        >>> probabilities = model.predict_proba(X)[:, 1]
        >>> predictions = model.predict(X)
    """
    logger.info(f"Loading curator model from {model_file}")
    model = joblib.load(model_file)
    logger.info("Curator model loaded successfully")
    return model


def train_curator_model(training_data_file: str, model_output_file: str) -> None:
    """
    Trains and calibrates a logistic regression model on curator selection data.

    Training pipeline:
    1. Load training and validation data
    2. Train LogisticRegression with CV sigmoid calibration on training set
    3. Evaluate on full validation set with comprehensive metrics
    4. Save calibrated model

    Uses 5-fold CV with sigmoid (Platt scaling) calibration which:
    - Preserves ROC-AUC while dramatically improving probability calibration
    - Works well with imbalanced/small datasets (only 2 parameters vs isotonic's N)
    - Uses all training data efficiently via cross-validation

    The model predicts the probability that a curator would select a given movie
    for broadcast, based on context features (time, day, etc.) and movie features
    (genre, ratings, etc.).

    Args:
        training_data_file: Path to joblib file containing training/validation data.
        model_output_file: Path where the calibrated model will be saved.
    """
    # Step 1: Load and prepare data
    X_train, y_train, X_val, y_val = _load_and_prepare_data(training_data_file)

    # Step 2: Train calibrated model using CV sigmoid calibration
    calibrated_model = _train_calibrated_model(X_train, y_train, cv=5)

    # Step 3: Evaluate on full validation set
    logger.info("Evaluating calibrated model on validation set...")
    y_pred = calibrated_model.predict(X_val)
    y_pred_proba = calibrated_model.predict_proba(X_val)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y_val, y_pred)
    logloss = log_loss(y_val, y_pred_proba, labels=[0, 1])
    roc_auc = roc_auc_score(y_val, y_pred_proba)
    brier = brier_score_loss(y_val, y_pred_proba)
    ece = _expected_calibration_error(y_val, y_pred_proba, n_bins=10)

    # Log metrics
    logger.info(f"\nValidation Accuracy@0.5: {accuracy:.4f}")
    logger.info(f"Validation ROC-AUC: {roc_auc:.4f}")
    logger.info(f"Validation Log Loss: {logloss:.4f}")
    logger.info(f"Validation Brier Score: {brier:.4f}")
    logger.info(f"Expected Calibration Error (ECE@10): {ece:.4f}")
    logger.info("Classification Report:\n" + classification_report(y_val, y_pred, zero_division=0))

    # Step 4: Save model
    logger.info(f"Saving trained and calibrated model to {model_output_file}")
    joblib.dump(calibrated_model, model_output_file)
    logger.info("Curator model training and calibration complete.")
