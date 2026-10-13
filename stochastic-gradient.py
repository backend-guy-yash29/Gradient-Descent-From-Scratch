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

class StochasticGradientDescent:
    def __init__(self,dataset):
        self.weights = []
        self.bias = 0
        self.dataset = dataset
        self.output = []


    def params_initializer(self):
        self.output = self.dataset['out'].to_numpy()
        self.dataset = self.dataset.drop('out',axis=1).to_numpy()

        self.weights = rng.random(self.dataset.shape[1])
        self.bias = rng.integers(low=1,high=10)

    def train(self,epochs=50,lr=1e-6,bias_lr=1e-3):
        self.params_initializer()
        n = len(self.dataset)
        for i in range(epochs+1):
            # shuffle so each epoch visits samples in a different order
            order = rng.permutation(n)

            for idx in order:
                x = self.dataset[idx]
                y = self.output[idx]

                # gradient from a single sample
                error  = x @ self.weights + self.bias - y
                grad_w = 2 * error * x
                grad_b = 2 * error

                self.weights -= lr * grad_w
                self.bias    -= bias_lr * grad_b

            if i % 10 == 0:
                y_pred = self.dataset @ self.weights + self.bias
                mse = np.mean((y_pred - self.output) ** 2)
                print(f"epoch {i:4d} | mse {mse:.4f}")
                print("Current Weights: ",self.weights,"Current Intercept: ",self.bias)

        print("Final Weights: ",self.weights,"Final Intercept: ",self.bias)


g1 = StochasticGradientDescent(dataset)
g1.train()
