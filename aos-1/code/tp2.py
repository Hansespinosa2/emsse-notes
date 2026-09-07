import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split


class BayesianLR:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.coef_ = None
        self.intercept_ = None
        self.sigma_sq = None
        self.beta_cov = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        x_aug = np.concat(
            [np.ones((X.shape[0], 1)), X],
            axis=1
        )

        XtX_inv = np.linalg.inv(x_aug.T @ x_aug)
        X_dag = XtX_inv @ x_aug.T
        w = X_dag @ y

        self.intercept_ = w[0]
        self.coef_ = w[1:]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.coef_ + self.intercept_


X, y = make_regression(
    n_samples=1000,
    n_features=1,
    n_informative=1,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42
)


def plot_regression(model, X_test, y_test):
    y_hat = model.predict(X_test)

    sort_idx = np.argsort(X_test[:, 0])
    X_sorted = X_test[sort_idx]
    y_hat_sorted = y_hat[sort_idx]

    plt.figure(figsize=(10, 6))
    plt.scatter(X_test[:, 0], y_test, alpha=0.6, label="Observed data")
    plt.plot(
        X_sorted[:, 0],
        y_hat_sorted,
        lw=2,
        label="Predicted outputs"
    )

    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    model = BayesianLR()
    model.fit(X_train, y_train)
    plot_regression(model, X_test, y_test)