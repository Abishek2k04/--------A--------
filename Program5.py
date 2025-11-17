
import numpy as np
class Perceptron:
    def __init__(self, n_inputs, lr=0.1, epochs=50):
        self.lr = lr
        self.epochs = epochs
        self.weights = np.zeros(n_inputs + 1)  # bias + weights

    def activation(self, x):
        return 1 if x >= 0 else 0

    def predict_raw(self, x):
        return np.dot(x, self.weights[1:]) + self.weights[0]

    def predict(self, x):
        return self.activation(self.predict_raw(x))

    def fit(self, X, y):
        for epoch in range(self.epochs):
            errors = 0
            for xi, target in zip(X, y):
                update = self.lr * (target - self.predict(xi))
                if update != 0:
                    errors += 1
                self.weights[1:] += update * xi
                self.weights[0] += update
            # optional: break early if no errors
            if errors == 0:
                break

def demo_or_xor():
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y_or = np.array([0,1,1,1])
    y_xor = np.array([0,1,1,0])

    p_or = Perceptron(n_inputs=2, lr=0.2, epochs=20)
    p_or.fit(X, y_or)
    print("OR gate results:")
    for xi in X:
        print(xi, "->", p_or.predict(xi))

    p_xor = Perceptron(n_inputs=2, lr=0.2, epochs=50)
    p_xor.fit(X, y_xor)
    print("\nXOR gate results (Perceptron cannot learn XOR reliably):")
    for xi in X:
        print(xi, "->", p_xor.predict(xi))

if __name__ == "__main__":
    demo_or_xor()