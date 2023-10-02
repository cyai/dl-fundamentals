import random
import pandas as pd
from loss_functions import Loss
from activation_functions import Activation

class Perceptron:
    def __init__(self, epochs, learning_rate, loss) -> None:
        self.epochs = 10
        self.learning_rate = 0.01
        if epochs:
            self.epochs = epochs
        if learning_rate:
            self.learning_rate = learning_rate
        
        self.loss = loss

    def dot_product(self, weights, inputs):
        return sum([w * x for w, x in zip(weights, inputs)])

    def read_csv(self, filename):
        df = pd.read_csv(filename)
        return df

    def input_output_split(self, df):
        inputs = df.iloc[:, :-1]
        outputs = df.iloc[:, -1]
        return inputs, outputs

    def init_weights(self, input_dim):
        weights = []
        for i in range(input_dim):
            weights.append(random.random())
        return weights

    def activation(self, x):
        return 1 if x >= 0 else 0

    def loss(self, z):
        loss = Loss(z) 
        if self.loss=='step':
            return loss.step()
        elif self.loss=='sigmoid':
            return loss.sigmoid()
        elif self.loss=='tanh':
            return loss.tanh()
        elif self.

    def train(self, inputs, outputs, weights, epochs):
        for epoch in range(epochs):
            for input, output in zip(inputs, outputs):
                prediction = self.activation(self.dot_product(weights, input))
                error = self.loss(output, prediction)
                weights = [
                    w + self.learning_rate * error * x for w, x in zip(weights, input)
                ]

            print(
                f"Epoch: {epoch} | Error: {error} | Prediction: {prediction} | Weights: {weights}"
            )
        return weights


if __name__ == "__main__":
    perceptron = Perceptron()
    df = perceptron.read_csv("data.csv")
    inputs, outputs = perceptron.input_output_split(df)
    weights = perceptron.init_weights(len(inputs.columns))
    weights = perceptron.train(
        inputs.values, outputs.values, weights, perceptron.epochs
    )
    print(weights)
