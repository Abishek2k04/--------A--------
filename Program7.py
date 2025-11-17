
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def plot_decision_boundary(clf, X, y, fname="svm_boundary.png"):
    x_min, x_max = X[:,0].min() - 0.5, X[:,0].max() + 0.5
    y_min, y_max = X[:,1].min() - 0.5, X[:,1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(8,6))
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:,0], X[:,1], c=y, edgecolor='k', s=50)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("SVM Decision Boundary (Iris - 2 features)")
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    print("Saved SVM decision boundary to", fname)

def main():
    iris = load_iris()
    # use first two features for 2D plotting
    X = iris.data[:, :2]
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    clf = SVC(kernel='linear', C=1.0, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("SVM accuracy (2-feature):", round(accuracy_score(y_test, y_pred), 4))

    try:
        plot_decision_boundary(clf, X, y)
    except Exception as e:
        print("Could not create plot (reason):", e)

if __name__ == "__main__":
    main()