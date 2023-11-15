from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from beartype import beartype
from beartype.typing import Literal, Optional, Sequence, Tuple, Union
from sklearn.base import BaseEstimator, is_classifier, is_regressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import KFold, LeaveOneOut, StratifiedKFold, train_test_split

from eis_toolkit import exceptions

SIMPLE_SPLIT = "simple_split"
KFOLD_CV = "kfold_cv"
SKFOLD_CV = "skfold_cv"
LOO_CV = "loo_cv"
NO_VALIDATION = "none"


@beartype
def save_model(model: BaseEstimator, path: Path) -> None:
    """
    Save a trained Sklearn model to a .joblib file.

    Args:
        model: Trained model.
        path: Path where the model should be saved. Include the .joblib file extension.
    """
    joblib.dump(model, path)


@beartype
def load_model(path: Path) -> BaseEstimator:
    """
    Load a Sklearn model from a .joblib file.

    Args:
        path: Path from where the model should be loaded. Include the .joblib file extension.

    Returns:
        Loaded model.
    """
    return joblib.load(path)


@beartype
def _train_and_evaluate_sklearn_model(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    model: BaseEstimator,
    test_method: Literal["simple_split", "kfold_cv", "skfold_cv", "loo_cv", "none"],
    metrics: Sequence[Literal["mse", "rmse", "mae", "r2", "accuracy", "precision", "recall", "f1"]],
    simple_split_size: float = 0.2,
    cv_folds: int = 5,
    random_state: Optional[int] = 42,
) -> Tuple[BaseEstimator, dict]:
    """
    Train and evaluate Sklearn model.

    Serves as a common private/inner function for Random Forest, Logistic Regression and Gradient Boosting
    public functions.
    """

    # Perform checks
    x_size = X.index.size if isinstance(X, pd.DataFrame) else X.shape[0]
    if x_size != y.size:
        raise exceptions.NonMatchingParameterLengthsException(f"X and y must have the length {x_size} != {y.size}.")
    if len(metrics) == 0 and test_method != NO_VALIDATION:
        raise exceptions.InvalidParameterValueException(
            "Metrics must have at least one chosen metric to validate model."
        )
    if cv_folds < 2:
        raise exceptions.InvalidParameterValueException("Number of cross-validation folds must be at least 2.")
    if not (0 < simple_split_size < 1):
        raise exceptions.InvalidParameterValueException("Test split must be more than 0 and less than 1.")

    # Approach 1: No validation
    if test_method == NO_VALIDATION:
        model.fit(X, y)
        metrics = {}

        return model, metrics

    # Approach 2: Simple split
    elif test_method == SIMPLE_SPLIT:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=simple_split_size, random_state=random_state
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        out_metrics = {}
        for metric in metrics:
            score = _score_model(model, y_test, y_pred, metric)
            out_metrics[metric] = score

    # Approach 3: Cross-validation
    elif test_method in [KFOLD_CV, SKFOLD_CV, LOO_CV]:
        cv = _get_cross_validator(test_method, cv_folds, random_state)

        # Initialize output metrics dictionary
        out_metrics = {}
        for metric in metrics:
            out_metrics[metric] = {}
            out_metrics[metric][f"{metric}_all"] = []

        # Loop over cross-validation folds and save metric scores
        for train_index, test_index in cv.split(X, y):
            model.fit(X[train_index], y[train_index])
            y_pred = model.predict(X[test_index])

            for metric in metrics:
                score = _score_model(model, y[test_index], y_pred, metric)
                all_scores = out_metrics[metric][f"{metric}_all"]
                all_scores.append(score)

        # Calculate mean and standard deviation for all metrics
        for metric in metrics:
            scores = out_metrics[metric][f"{metric}_all"]
            out_metrics[metric][f"{metric}_mean"] = np.mean(scores)
            out_metrics[metric][f"{metric}_std"] = np.std(scores)

        # Fit on entire dataset after cross-validation
        model.fit(X, y)

        # If we calculated only 1 metric, remove the outer dictionary layer from output
        if len(out_metrics) == 1:
            out_metrics = out_metrics[metrics[0]]

    else:
        raise Exception(f"Unrecognized test method: {test_method}")

    return model, out_metrics


@beartype
def _score_model(
    model: BaseEstimator,
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    metric: Literal["mse", "rmse", "mae", "r2", "accuracy", "precision", "recall", "f1"],
) -> float:
    """Score a Sklearn model's predictions using the selected metric."""

    if metric in ["mae", "mse", "rmse", "r2"] and not is_regressor(model):
        raise exceptions.InvalidParameterValueException(
            f"Chosen metric ({metric}) is not applicable for given model type (classifier)."
        )
    if metric in ["accuracy", "precision", "recall", "f1"] and not is_classifier(model):
        raise exceptions.InvalidParameterValueException(
            f"Chosen metric ({metric}) is not applicable for given model type (regressor)."
        )

    if is_classifier(model):
        if len(y_true) > 2:  # Multiclass prediction
            average_method = "micro"
        else:  # Binary prediction
            average_method = "binary"

    if metric == "mae":
        score = mean_absolute_error(y_true, y_pred)
    elif metric == "mse":
        score = mean_squared_error(y_true, y_pred)
    elif metric == "rmse":
        score = mean_squared_error(y_true, y_pred, squared=False)
    elif metric == "r2":
        score = r2_score(y_true, y_pred)
    elif metric == "accuracy":
        score = accuracy_score(y_true, y_pred)
    elif metric == "precision":
        score = precision_score(y_true, y_pred, average=average_method)
    elif metric == "recall":
        score = recall_score(y_true, y_pred, average=average_method)
    elif metric == "f1":
        score = f1_score(y_true, y_pred, average=average_method)
    else:
        raise exceptions.InvalidParameterValueException(f"Unrecognized metric: {metric}")

    return score


@beartype
def _get_cross_validator(
    cv: str, folds: int, random_state: Optional[int]
) -> Union[KFold, StratifiedKFold, LeaveOneOut]:
    """Create and return a Sklearn cross-validator based on given parameter values."""
    if cv == KFOLD_CV:
        cross_validator = KFold(n_splits=folds, shuffle=True, random_state=random_state)
    elif cv == SKFOLD_CV:
        cross_validator = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    elif cv == LOO_CV:
        cross_validator = LeaveOneOut()
    else:
        raise exceptions.InvalidParameterValueException(f"CV method was not recognized: {cv}")

    return cross_validator
