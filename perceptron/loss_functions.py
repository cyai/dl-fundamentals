import math
import numpy as np


class Loss:
    def __init__(self, y, y_hat, loss) -> None:
        self.y = y
        self.y_hat = y_hat
        self.loss_function = loss

    def loss(self):
        if self.loss_function=='mse':
            return self.mean_square_error()
        elif self.loss_function=='mae':
            return self.mean_absolute_error()
        elif self.loss_function=='binary_cross_entropy':
            return self.binary_cross_etropy()
        elif self.loss_function=='huber_loss':
            return self.huber_loss()
        elif self.loss_function=='categorical_binary_cross_entropy':
            return self.categorical_binary_cross_entropy()
        else:
            raise ValueError("Loss function not supported")
        
    def mean_square_error(self):
        return (self.y - self.y_hat) ** 2

    def mean_absolute_error(self):
        return self.y - self.y_hat

    def binary_cross_etropy(self):
        return self.y * math.log(self.y_hat) - (1 - self.y) * math.log(1 - self.y_hat)

    def huber_loss(self, delta):
        return (
            0.5 * (self.y - self.y_hat) ** 2
            if (math.abs(self.y - self.y_hat) <= delta)
            else delta * (math.abs(self.y - self.y_hat) - 0.5 * (delta**2))
        )

    def categorical_binary_cross_entropy(self, n):
        """
        n ---> Number of classes in output layer
        """
        return -np.sum(self.y * np.log(self.y_hat + 10**-100))
