
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

def main():
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    nb = GaussianNB()
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("GaussianNB Accuracy:", round(acc, 4))
    print("Confusion Matrix:\n", cm)
    print("\nPer-class precision-like check (TP / predicted):")
    for i in range(len(cm)):
        tp = cm[i, i]
        preds = cm[:, i].sum()
        prec = tp / preds if preds != 0 else 0.0
        print(f"  Class {i}: approx precision {prec:.3f}")

if __name__ == "__main__":
    main()