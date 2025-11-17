
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def main():
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = DecisionTreeClassifier(random_state=42, max_depth=4)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print("Decision Tree accuracy:", round(acc, 4))
    print("Feature importances:", list(zip(iris.feature_names, clf.feature_importances_)))

    # Plot and save the tree to a file
    try:
        plt.figure(figsize=(10,6))
        plot_tree(clf, feature_names=iris.feature_names, class_names=iris.target_names, filled=True, rounded=True)
        plt.tight_layout()
        plt.savefig("decision_tree_iris.png", dpi=150)
        print("Saved tree visualization to decision_tree_iris.png")
    except Exception as e:
        print("Could not plot tree (reason):", e)

if __name__ == "__main__":
    main()