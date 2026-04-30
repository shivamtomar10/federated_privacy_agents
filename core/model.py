import numpy as np


class FederatedModel:
    """
    Multi-Class Federated Model (classification)
    --------------------------------------------
    - supports DP-friendly gradient descent
    - output: class scores (not regression)
    - weights shape: (features, classes)
    """

    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim

        # initialize weights for all classes
        if input_dim == 0:
            self.weights = np.zeros((1, output_dim))
        else:
            self.weights = np.random.randn(input_dim, output_dim) * 0.01

    def train(self, X, y, lr=0.01, epochs=5):
        """
        X: (n_samples, features)
        y: (n_samples,) with class labels
        """

        n_samples = len(y)

        # one-hot encoding for gradient
        one_hot = np.eye(self.output_dim)[y]

        for _ in range(epochs):

            # logits
            logits = X @ self.weights

            # softmax
            exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs = exp / np.sum(exp, axis=1, keepdims=True)

            # gradient
            grad = (X.T @ (probs - one_hot)) / n_samples

            # update
            self.weights -= lr * grad

        return self.weights

    def evaluate(self, X, y):
        """
        Returns accuracy
        """

        if X.shape[1] == 0:
            preds = np.zeros(len(y))
        else:
            logits = X @ self.weights
            preds = np.argmax(logits, axis=1)

        return np.mean(preds == y)
<<<<<<< HEAD
=======

>>>>>>> 19b0456 (Initial federated privacy agents code)
