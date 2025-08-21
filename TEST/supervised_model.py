from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional
import numpy as np

def train_sklearn(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.25,
    random_state: Optional[int] = 42,
    verbose: bool = False
) -> Tuple[LogisticRegression, float]:
    """
    Train a Logistic Regression classifier using scikit-learn.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.
        test_size (float): Fraction of data to use as test set. Default is 0.25.
        random_state (int, optional): Random seed for reproducibility.
        verbose (bool): If True, prints training progress.

    Returns:
        Tuple[LogisticRegression, float]: Trained model and accuracy on test set.
    """
    if len(np.unique(y)) < 2:
        raise ValueError("Training labels must contain at least two classes.")

    stratify = y if len(y) >= 20 else None  # Only stratify if dataset large enough

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    model = LogisticRegression(multi_class='auto', max_iter=1000)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    if verbose:
        print(f"[SupervisedModel] Training complete. Test accuracy: {accuracy:.4f}")

    return model, accuracy
