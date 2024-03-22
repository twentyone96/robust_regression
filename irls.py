import numpy as np

'''
@brief:
    Iteratively reweighted least squares (IRLS) algorithm
@param:
    X: Design matrix
    y: Target vector
    beta0: Initial guess of coefficients
    weight_func: Weight function
    max_iters: Maximum number of iterations
@return:
    beta: coefficients
    weights: Weights
    num_iters: Number of iterations
'''
def irls(X, y, beta0, weight_func, max_iters=100):
    beta = beta0.copy()
    condition = TerminationCondition(beta, max_iters)

    while not condition(beta):
        Psi = np.diag(weight_func(y - X @ beta))
        beta = np.linalg.inv(X.T @ Psi @ X) @ X.T @ Psi @ y
    num_iters = condition.curr_iters
    weights = Psi.diagonal()
    return beta, weights, num_iters


'''
@brief:
    Termination condition for IRLS
@param:
    beta0: Initial guess of coefficients
    max_iters: Maximum number of iterations
    tol: Tolerance
@return:
    True if the algorithm should terminate, False otherwise
'''
class TerminationCondition:
    def __init__(self, beta, max_iters=100, tol=1e-6):
        self.beta_prev = beta.copy() + tol*2
        self.curr_iters = 0
        self.max_iters = max_iters
        self.tol = tol
    
    def __call__(self, beta_curr):
        self.curr_iters += 1
        if self.curr_iters >= self.max_iters:
            return True
        elif np.linalg.norm(beta_curr - self.beta_prev) <= self.tol:
            return True
        else:
            self.beta_prev = beta_curr.copy()
            return False