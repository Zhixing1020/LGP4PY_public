"""The :mod:`lgp` module provides the necessary methods and classes to perform
Linear Genetic Programming (LGP). It provides necessary functions run LGP.
The behaviors of LGP is defined by a parameter file.

This file is equivalent to "Evolve.java" in ECJ
"""



# import classes and files
import sys
import os

# Get current working directory
cwd = os.getcwd()

# Insert at the front of sys.path (highest priority)
if cwd not in sys.path:
    sys.path.insert(0, cwd)

from src.ec import *
from src.ec.util import *
from tasks.symbreg.lgp_regressor import LinearGP_Regressor
from tasks.classification.lgp_classifier import LinearGP_Classifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_gaussian_quantiles
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score


def test_SR():
    """Test LGP for solving symbolic regression."""

    # Example data: y = X^6 + X^5 + X^4 + X^3 + X^2 + X + X_2^2
    X = np.random.rand(100, 1) * 2 - 1
    X_2 = X + 4
    y = X**6 + X**5 + X**4 + X**3 + X**2 + X + X_2**2

    X_train, X_test, y_train, y_test = train_test_split(
        np.column_stack([X, X_2]), y, test_size=0.5
    )

    cwd = os.getcwd()
    param_file = f"{cwd}/tasks/symbreg/algorithm/FunctionScaling/parameters/LGP_numpy_FuncScale_SRBench.params"
    # param_file = f"{cwd}/tasks/symbreg/algorithm/FunctionScaling/parameters/LGP_FuncScale_SRBench.params"
    # param_file = f"{cwd}/tasks/symbreg/algorithm/FunctionScaling/parameters/LGP_SR_BasicType.params"
    # param_file = f"{cwd}/tasks/symbreg/algorithm/FunctionScaling/parameters/LGP_SR_EDA_dsl.params"
    # param_file = f"{cwd}/tasks/symbreg/parameters/LGP_test.params"

    lgp = LinearGP_Regressor(param_file=param_file)
    lgp.fit(X_train, y_train)

    pred = lgp.predict(X_test)

    print(f"test R2: {r2_score(y_test, pred)}")


def test_Classification(dataset_name: str = "linear3"):
    """Test LGP for classification on selectable 3-class 2D tasks.

    dataset_name options:
    - "linear3": synthetic 3-class data from linear thresholds
    - "gaussianq3": sklearn make_gaussian_quantiles 3-class non-linear dataset
    """

    np.random.seed(42)
    n = 120

    if dataset_name == "linear3":
        X = np.random.rand(n, 2) * 4 - 2
        score = X[:, 0] + X[:, 1]
        y = np.zeros(n, dtype=float)
        y[score > -0.5] = 1.0
        y[score > 0.8] = 2.0
        dataset_title = "Synthetic Linear 3-Class"
    elif dataset_name == "gaussianq3":
        X, y_int = make_gaussian_quantiles(
            n_samples=n,
            n_features=2,
            n_classes=3,
            cov=1.8,
            random_state=42,
        )
        y = y_int.astype(float)
        dataset_title = "sklearn make_gaussian_quantiles 3-Class (Non-Linear)"
    else:
        raise ValueError("dataset_name must be 'linear3' or 'gaussianq3'")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    cwd = os.getcwd()
    param_file = f"{cwd}/tasks/classification/parameters/LGP_classification_test.params"

    lgp = LinearGP_Classifier(param_file=param_file)
    lgp.fit(X_train, y_train)

    pred = np.asarray(lgp.predict(X_test), dtype=float)

    print(f"[{dataset_name}] test accuracy: {accuracy_score(y_test, pred):.4f}")

    # Plot train/test samples with true labels and highlight misclassified test points.
    y_train_np = np.asarray(y_train, dtype=float)
    y_test_np = np.asarray(y_test, dtype=float)
    wrong_mask = pred != y_test_np

    class_labels = np.unique(np.concatenate([y_train_np, y_test_np]))
    cmap = plt.cm.get_cmap("tab10", len(class_labels))

    fig, ax = plt.subplots(figsize=(8, 6))

    for i, label in enumerate(class_labels):
        train_mask = y_train_np == label
        test_mask = y_test_np == label

        ax.scatter(
            X_train[train_mask, 0],
            X_train[train_mask, 1],
            c=[cmap(i)],
            marker="o",
            s=42,
            alpha=0.65,
            label=f"Train class {label:g}",
        )
        ax.scatter(
            X_test[test_mask, 0],
            X_test[test_mask, 1],
            c=[cmap(i)],
            marker="^",
            s=64,
            edgecolors="black",
            linewidths=0.6,
            alpha=0.95,
            label=f"Test class {label:g}",
        )

    if np.any(wrong_mask):
        ax.scatter(
            X_test[wrong_mask, 0],
            X_test[wrong_mask, 1],
            marker="x",
            c="red",
            s=140,
            linewidths=2.2,
            label="Wrong test prediction",
            zorder=10,
        )

    ax.set_title(
        f"LGP Classification ({dataset_title}): Train/Test Labels and Misclassified Test Samples"
    )
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # test_SR()
    # test_Classification(dataset_name="linear3")
    test_Classification(dataset_name="gaussianq3")

    # Evolve.main()