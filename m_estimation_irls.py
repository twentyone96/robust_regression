import numpy as np
import matplotlib.pyplot as plt

from weight_func import L2Weight, HuberWeight, TukeyWeight, CorrentropyWeight
from irls import irls

def random_noise(num_points, outlier_fraction=0.2):
    n_outliers = int(outlier_fraction * num_points)
    noise_arr = np.random.randn(num_points)
    outliers_index = np.random.choice(np.arange(num_points), n_outliers, replace=False)
    noise_arr[outliers_index] = 20 * np.random.rand(n_outliers)
    return noise_arr

if __name__ == "__main__":
    np.random.seed(2021)

    num_points = 20
    outlier_fraction = 0.2
    beta = np.array([1, 2])

    x_truth = np.vstack([np.linspace(0, 1, num_points), np.ones(num_points)])
    y_truth = x_truth.T @ beta
    x_meas = x_truth.copy()
    y_meas = y_truth + random_noise(num_points, outlier_fraction)

    beta_init = np.array([-1, -1])
    beta_l2,    weight_l2,     num_iters_l2 = irls(x_meas.T, y_meas, beta_init, L2Weight())
    beta_huber, weight_huber,  num_iters_huber = irls(x_meas.T, y_meas, beta_init, HuberWeight(k=0.6))
    beta_tukey, weight_tukey,  num_iters_tukey = irls(x_meas.T, y_meas, beta_init, TukeyWeight(k=6.0))
    beta_corr,  weight_corr,   num_iters_corr = irls(x_meas.T, y_meas, beta_init, CorrentropyWeight(sigma=2.5))

    print("Truth", beta)
    print("L2:", beta_l2)
    print("Huber:", beta_huber)
    print("Tukey:", beta_tukey)
    print("Correntropy:", beta_corr)

    print("Number of iterations (L2):", num_iters_l2)
    print("Number of iterations (Huber):", num_iters_huber)
    print("Number of iterations (Tukey):", num_iters_tukey)
    print("Number of iterations (Correntropy):", num_iters_corr)
    
    plt.figure(dpi=150)
    plt.plot(x_meas[0,:], y_meas, 'k.')
    plt.plot(x_truth[0,:], y_truth, 'r-')
    plt.plot(x_truth[0,:], x_truth.T @ beta_l2, 'b--', label='L2')
    plt.plot(x_truth[0,:], x_truth.T @ beta_huber, 'm--', label='Huber')
    plt.plot(x_truth[0,:], x_truth.T @ beta_tukey, 'c--', label='Tukey')
    plt.plot(x_truth[0,:], x_truth.T @ beta_corr, 'y--', label='Correntropy')
    plt.title("M Estimation (IRLS)")
    plt.legend()
    plt.show()