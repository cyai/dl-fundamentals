import math
import numpy as np


class Loss:
    def __init__(self, y, y_hat) -> None:
        self.y = y
        self.y_hat = y_hat

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
