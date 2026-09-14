import numpy as np
import scipy as sp
import scipy.stats as spst
import matplotlib.pyplot as plt

### Part 1
def f_latent(x):
    return x * np.cos(x)

def generate_and_plot_data(n_size: int, plot: bool = False):
    x = np.random.uniform(0, 100, n_size)
    y = f_latent(x)
    if plot:
        plt.scatter(x, y)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()
    return x, y


### Part 2
def se_kernel(X_1, X_2, sigma_f_squared=1.0, l=1.0):
    X_1 = np.asarray(X_1).reshape(-1, 1)
    X_2 = np.asarray(X_2).reshape(-1, 1)

    squared_distances = (X_1 - X_2.T)**2

    K = sigma_f_squared * np.exp(
        -squared_distances / (2 * l**2)
    )

    return K
### Part 3
def pred_dist(Xstar, X, y, sigma_n=1e-10, l=1, sigma_f=1):
    if len(Xstar.shape) == 1:
        Xstar = Xstar.reshape(-1,1)

