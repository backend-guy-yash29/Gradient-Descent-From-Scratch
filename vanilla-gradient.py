import numpy as np
import pandas as pd


# generating synthetic dataset
rng = np.random.default_rng()
random_arr = rng.random(7000)

inp_col = np.linspace(start=-500,stop=500,num=7000)
out_col = 3.5 * inp_col + 2.5 + random_arr

dataset = pd.DataFrame({'inp':inp_col,'out':out_col})
# print("Dataframe: ",dataset.head(20))
# dataset.info()

class GradientDescent:
    def __init__(self,dataset):
        self.weights = []
        self.bias = 0
        self.dataset = dataset
        self.output = []
        

    def params_initializer(self):
        self.output = self.dataset['out']
        self.dataset = self.dataset.drop('out',axis=1)

        self.weights = rng.random(len(self.dataset.columns))
        self.bias = rng.integers(low=1,high=10)

    def train(self,epochs=10000,lr=1e-6,bias_lr=1e-3):
        self.params_initializer()
        history = []
        for i in range(epochs+1):
            y_pred = self.dataset @ self.weights + self.bias
            error  = y_pred - self.output
            
            if i % 1000 == 0:
                mse = np.mean(error ** 2)
                print(f"epoch {i:4d} | mse {mse:.4f}")
                print("Current Weights: ",self.weights,"Current Intercept: ",self.bias)


            n = len(self.dataset)
            grad_w = (2 / n) * (self.dataset.T @ error)
            grad_b = (2 / n) * np.sum(error)

            self.weights -= lr * grad_w
            self.bias    -= bias_lr * grad_b
            

            
        print("FInal Weights: ",self.weights,"Final Intercept: ",self.bias)


g1 = GradientDescent(dataset)
# g1.params_initializer()
g1.train()

    