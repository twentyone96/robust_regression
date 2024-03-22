import numpy as np
import matplotlib.pyplot as plt

class WeightBase:
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
    
class HuberWeight(WeightBase):
    def __init__(self, k=1.345):
        self.k = k
    
    def func(self, e):
        if abs(e) <= self.k:
            return 1.0
        else:
            return self.k / abs(e)
        
class TukeyWeight(WeightBase):
    def __init__(self, k=4.685):
        self.k = k
    
    def func(self, e):
        if abs(e) <= self.k:
            return (1 - (e / self.k)**2)**2
        else:
            return 0
        
class CorrentropyWeight(WeightBase):
    def __init__(self, sigma=2.1105):
        self.sigma = sigma
        
    def func(self, e):
        return 1 / (np.power(self.sigma,3) * np.sqrt(2*np.pi) ) * np.exp(-0.5*np.power(e/self.sigma, 2))
        
class L2Weight(WeightBase):
    def func(self, e):
        return 1.0
    
if __name__ == "__main__":
    x = np.linspace(-6, 6, 100)
    plt.figure(dpi=150)
    plt.title("Weight Function")
    plt.xlabel("Residual")
    plt.ylabel("Weight")
    plt.plot(x, L2Weight()(x), 'b-', label='L2')
    plt.plot(x, HuberWeight()(x), 'm-', label='Huber')
    plt.plot(x, TukeyWeight()(x), 'c-', label='Tukey')
    plt.plot(x, 23.6*CorrentropyWeight()(x), 'y-', label='Correntropy (scaled by 23.6)')
    plt.legend()
    plt.show()