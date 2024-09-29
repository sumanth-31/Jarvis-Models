from abc import abstractmethod,ABC
from IPython.display import clear_output

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, accuracy_score


class UnivariateRegression(ABC):
    weights = None
    n_iterations = None
    learning_rate = None
    dataset = None
    input_features = None
    target_feature = None
    m = None
    n = None
    def __init__(self, dataset , n_iterations = 1000, learning_rate = 0.01, weights = None):
        self.dataset = dataset
        pre_processed_dataset = self.pre_process(self.dataset.copy())
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.input_features = pre_processed_dataset.iloc[:, :-1].copy()
        self.target_feature = pre_processed_dataset.iloc[:, -1].copy()
        self.m = self.input_features.shape[0]
        self.n = self.input_features.shape[1]
        if (self.weights == None):
            self.weights = pd.Series([0]*self.n)
        print('input_features = ', self.input_features)
        print('target features = ', self.target_feature)
        print('weights = ', self.weights)
    def pre_process(self, dataset):
        dataset.dropna(inplace = True)
        dataset.reset_index(drop=True, inplace=True)
        # Scale Features
        standard_scaler = StandardScaler()
        dataset = pd.DataFrame(standard_scaler.fit_transform(dataset))
        # Add intercept term to the features
        intercept_feature = pd.Series([1]*dataset.shape[0])
        dataset = pd.concat([intercept_feature, dataset], axis = 1)
        return dataset
    def update_weight(self, index, gradient):
        self.weights.iloc[index] -= self.learning_rate*gradient
    @abstractmethod
    def calculate_cost(self):
        pass
    @abstractmethod
    def calculate_hypothesis(self, input_features):
        pass
    def calculate_gradient(self,feature_index):
        hypothesis = self.calculate_hypothesis(self.input_features)
        xj = self.input_features.iloc[:, feature_index]
        h_y_term = hypothesis - self.target_feature
        sum_term = np.dot(h_y_term.values, xj.values).item()
        return (1/self.m)*sum_term
    def train(self, verbose_costs = True):
        costs = []
        for i in range(self.n_iterations):
            clear_output(wait=True)
            print('Calculating gradients for iteration ', i)
            for j in range(self.n):
                gradient = self.calculate_gradient(j)
                self.update_weight(j, gradient)
            if (verbose_costs):
                cost = self.calculate_cost()
                costs.append(cost)
                print('Cost after', i, ' iterations = ', cost)
            print('Weights after', i, ' iterations = ', self.weights)
        if (verbose_costs):
            plt.plot(range(len(costs)), costs)
        return self.weights
    def calculate_accuracy_r2(self, y_actual, y_predicted):
        return r2_score(y_actual, y_predicted)
    def calculate_accuracy_classification(self, y_actual, y_predicted):
        return accuracy_score(y_actual, y_predicted)
    def predict(self, input_features):
        input_features_with_intercept = pd.concat([pd.Series([1]*input_features.shape[0]), input_features], axis = 1)
        return self.calculate_hypothesis(input_features_with_intercept)
    def plot_trained_model(self, input_features, target_feature):
        y_predicted = self.predict(input_features)
        y_actual_scatter = plt.scatter(input_features, target_feature, color = 'blue', marker='o')
        y_predicted_scatter = plt.scatter(input_features, y_predicted, color = 'red', marker='x')
        plt.legend((y_actual_scatter, y_predicted_scatter), ('Actual', 'Predicted'))
        plt.plot(input_features, y_predicted)
        plt.show()