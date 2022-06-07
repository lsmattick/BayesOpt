from scipy.stats import binom, beta
import numpy as np


class GaussianProcess():
    def __init__(self, x_data: list, y_data: list = None):
        self.x_data = x_data

    @staticmethod
    def exponential_cov(x, y, params):
        return params[0] * np.exp(-0.5 * params[1] * np.subtract.outer(x, y)**2)

    @staticmethod
    def conditional(x_new, x, y, params):
        Sigma_x_new_x = __class__.exponential_cov(x_new, x, params)
        Sigma_x_x_new = __class__.exponential_cov(x, x_new, params)
        Sigma_x_x = __class__.exponential_cov(x, x, params)
        Sigma_x_new_x_new = __class__.exponential_cov(x_new, x_new, params)

        Sigma_x_inv = np.linalg.inv(Sigma_x_x)

        mu = 0 + np.dot(Sigma_x_new_x, Sigma_x_inv).dot(y)
        sigma = Sigma_x_new_x_new - np.dot(Sigma_x_new_x, Sigma_x_inv).dot(Sigma_x_x_new)
        return(mu.squeeze(), sigma.squeeze())
