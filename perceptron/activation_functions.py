import math
from typing import Any


class Activation:
    def __init__(self, z, activation) -> None:
        self.z = z
        self.activation_function = activation
    
    def activation(self):
        if self.activation_function=='step':
            return self.step()
        elif self.activation_function=='sigmoid':
            return self.sigmoid()
        elif self.activation_function=='tanh':
            return self.tanh()
        elif self.activation_function=='relu':
            return self.relu()
        elif self.activation_function=='leaky_relu':
            return self.leaky_relu()
        elif self.activation_function=='elu':
            return self.elu()
        else:
            raise ValueError("Activation function not supported")

    def step(self):
        return 1 if self.z >= 0 else 0

    def sigmoid(self):
        return 1 / (1 + math.exp(-self.z))

    def tanh(self):
        return (math.exp(self.z) - math.exp(-self.z)) / (
            math.exp(self.z) + math.exp(-self.z)
        )

    def relu(self):
        return max(0, self.z)

    def leaky_relu(self):
        return max(0.1 * self.z, self.z)

    def elu(self):
        return self.z if self.z >= 0 else 0.1 * (math.exp(self.z) - 1)
