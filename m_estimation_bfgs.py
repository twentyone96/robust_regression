import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

from score_func import L2Score, L1Score, HuberScore, TukeyScore, CorrentropyScore

def cost_func(w, x, y, score_func):
    residual = y - x @ w
    return np.sum([score_func(r) for r in residual])

def minimize(X, y, beta0, score_func=L2Score(), method='BFGS'):
    res = scipy.optimize.minimize(cost_func, beta0, args=(X, y, score_func), method=method)
    return res
        
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
    res_l2 = minimize(x_meas.T, y_meas, beta_init, L2Score())
    res_l1 = minimize(x_meas.T, y_meas, beta_init, L1Score())
    res_huber = minimize(x_meas.T, y_meas, beta_init, HuberScore(k=0.6))
    res_tukey = minimize(x_meas.T, y_meas, beta_init, TukeyScore(k=6.0))
    res_correntropy = minimize(x_meas.T, y_meas, beta_init, CorrentropyScore(sigma=2.5))

    beta_l2 = res_l2.x
    beta_l1 = res_l1.x
    beta_huber = res_huber.x
    beta_tukey = res_tukey.x
    beta_correntropy = res_correntropy.x
    print("Truth", beta)
    print("L2:", beta_l2)
    print("L1:", beta_l1)
    print("Huber:", beta_huber)
    print("Tukey:", beta_tukey)
    print("Correntropy:", beta_correntropy)

    plt.figure(dpi=150)
    plt.plot(x_meas[0,:], y_meas, 'k.')
    plt.plot(x_truth[0,:], y_truth, 'r-')
    plt.plot(x_truth[0,:], x_truth.T @ beta_l2, 'b--', label='L2')
    plt.plot(x_truth[0,:], x_truth.T @ beta_l1, 'g--', label='L1')
    plt.plot(x_truth[0,:], x_truth.T @ beta_huber, 'm--', label='Huber')
    plt.plot(x_truth[0,:], x_truth.T @ beta_tukey, 'c--', label='Tukey')
    plt.plot(x_truth[0,:], x_truth.T @ beta_correntropy, 'y--', label='Correntropy')
    plt.title("M Estimation (BFGS)")
    plt.legend()
    plt.show()