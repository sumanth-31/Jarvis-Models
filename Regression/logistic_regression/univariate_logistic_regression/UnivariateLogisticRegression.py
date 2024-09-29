import numpy as np
import pandas as pd
from math import exp

from regression import UnivariateRegression

class UnivariateLogisticRegression(UnivariateRegression):
    def __init__(self, dataset , n_iterations = 1000, learning_rate = 0.01, weights = None):
        super().__init__(dataset, n_iterations, learning_rate, weights)
    def pre_process(self, dataset):
        target_feature_name = dataset.columns[-1]
        target_feature = dataset[target_feature_name]
        dataset.dropna(inplace = True)
        dataset = pd.get_dummies(dataset, drop_first = True)
        dataset = dataset.drop(target_feature_name, axis = 1)
        dataset[target_feature_name] = target_feature
        pre_processed_dataset = super().pre_process(dataset)
        pre_processed_dataset.iloc[:,-1] = target_feature # To avoid scaling target_feature so that log_loss works as expected
        return pre_processed_dataset
    def calculate_cost(self):
        hypothesis = self.calculate_hypothesis(self.input_features)
        log_loss_first_term = self.target_feature * np.log(hypothesis)
        log_loss_second_term = (1 - self.target_feature) * np.log(1 - hypothesis)
        log_loss = np.sum(log_loss_first_term) + np.sum(log_loss_second_term)
        return -(1/self.m) * log_loss
    def calculate_hypothesis(self, input_features):
        linear_equation = np.dot(input_features, self.weights)
        res = 1/(1 + np.exp(-linear_equation))
        return res
    def predict(self, input_features):
        hypothesis_result = super().predict(input_features)
        return np.where(hypothesis_result >= 0.5, 1, 0)
