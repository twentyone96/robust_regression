import numpy as np
import matplotlib.pyplot as plt

class ScoreBase:
    def __init__(self):
        pass

    def __call__(self, x):
        if isinstance(x, list) or isinstance(x, tuple) or isinstance(x, np.ndarray):
            data_list = np.array(x).tolist()
            return np.array([self.func(e) for e in data_list])
        elif isinstance(x, float) or isinstance(x, int):
            return self.func(x)
        else:
            raise TypeError("Input type is not supported")
    
    def func(self, e):
        raise NotImplementedError

class HuberScore(ScoreBase):
    def __init__(self, k=1.345):
        self.k = k
        
    def func(self, e):
        if abs(e) <= self.k:
            return 0.5*pow(e, 2)
        else:
            return self.k*np.abs(e) - 0.5*pow(self.k, 2)
        
class TukeyScore(ScoreBase):
    def __init__(self, k=4.685):
        self.k = k
        self.c = self.k**2 / 6
        
    def func(self, e):
        if abs(e) <= self.k:
            return self.c*(1 - (1 - (e/self.k)**2)**3)
        else:
            return self.c

class CorrentropyScore(ScoreBase):
    def __init__(self, sigma=2.1105):
        self.sigma = sigma
        
    def func(self, e):
        return 1 / (self.sigma*np.sqrt(2*np.pi)) * (1 - np.exp(-0.5*pow(e/self.sigma, 2)))

class L2Score(ScoreBase):
    def func(self, e):
        return 0.5*pow(e, 2)
    
class L1Score(ScoreBase):
    def func(self, e):
        return abs(e)

if __name__ == "__main__":
    x = np.linspace(-6, 6, 100)
    plt.figure(dpi=150)
    plt.title("Score Function")
    plt.xlabel("Residual")
    plt.ylabel("Score")
    plt.plot(x, L2Score()(x), 'b-', label='L2')
    plt.plot(x, L1Score()(x), 'g-', label='L1')
    plt.plot(x, HuberScore()(x), 'm-', label='Huber')
    plt.plot(x, TukeyScore()(x), 'c-', label='Tukey')
    plt.plot(x, 50*CorrentropyScore()(x), 'y-', label='Correntropy (scaled by 50)')
    plt.legend()
    plt.show()