import numpy as np

from regression import UnivariateRegression

class UnivariateLinearRegression(UnivariateRegression):
    def __init__(self, dataset , n_iterations = 1000, learning_rate = 0.01, weights = None):
        super().__init__(dataset, n_iterations, learning_rate, weights)
    def calculate_cost(self):
        residuals = self.calculate_hypothesis(self.input_features) - self.target_feature
        residual_squared = residuals ** 2
        residual_squared_sum = np.sum(residual_squared)
        return (1/(2*self.m)) * residual_squared_sum
    def calculate_hypothesis(self, input_features):
        return np.dot(input_features, self.weights)
