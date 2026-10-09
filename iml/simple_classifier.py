from matplotlib.colors import BoundaryNorm
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class SimpleClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, n_classes: int):
        self.n_classes = n_classes

    def predict_proba(self, x):
        logits = x @ self.weights_ + self.bias_  # (b, c)
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, x):
        probs = self.predict_proba(x)
        return np.argmax(probs, axis=1)

    def fit(self, x, y, lr, steps=100):
        # x has shape (b, d): it's a matrix of b datapoints, each with d dimensions
        # y has shape (b,): it's an list of classes, represented as integers from 0 to n_classes - 1
        raise NotImplementedError  # you should override this method!

def show_iris_classifier(model):
    """Show decision boundaries and labelled datapoints for a classifier trained on
    the iris dataset using sepal length and width as the two input features."""
    iris = load_iris()
    x = iris.data[:, :2]  # sepal length, sepal width
    y = iris.target
    names = iris.target_names

    sns.set_theme()
    fig, ax = plt.subplots()
    fig.set_size_inches(7, 5)
    ax.set_xlabel("Sepal length (cm)")
    ax.set_ylabel("Sepal width (cm)")
    ax.set_title("Iris classification")

    _show_classifier(ax, x, y, names, model)
    plt.tight_layout()
    plt.show(fig)


def _show_classifier(ax, x, y, class_names, model):
    k = len(class_names)
    margin = 0.3
    x_min, x_max = x[:, 0].min() - margin, x[:, 0].max() + margin
    y_min, y_max = x[:, 1].min() - margin, x[:, 1].max() + margin

    res = 300
    x0, x1 = np.meshgrid(
        np.linspace(x_min, x_max, num=res),
        np.linspace(y_min, y_max, num=res),
    )

    x_mesh = np.stack([x0, x1], axis=-1).reshape(-1, 2)
    probs = model.predict_proba(x_mesh).reshape(res, res, -1)
    classes = np.argmax(probs, axis=-1)

    cmap = plt.get_cmap("tab10")
    ax.pcolormesh(
        x0, x1, classes,
        cmap=cmap,
        norm=BoundaryNorm(np.arange(k + 1), k),
        alpha=0.35,
        shading="auto",
    )

    for cls_idx, name in enumerate(class_names):
        mask = y == cls_idx
        ax.scatter(
            x[mask, 0], x[mask, 1],
            color=cmap(cls_idx),
            edgecolors="white",
            linewidths=0.6,
            label=name,
            zorder=3,
        )

    ax.legend(title="Species", framealpha=0.8)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
