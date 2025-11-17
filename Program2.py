# program2_regression_and_multivariate_regression.py
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

def univariate_regression(X, y, feature_index=2, test_size=0.2, random_state=42):
    Xf = X[:, feature_index].reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(Xf, y, test_size=test_size, random_state=random_state)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    print("Univariate Linear Regression (feature index={}):".format(feature_index))
    print("  Coef: {:.4f}, Intercept: {:.4f}".format(lr.coef_[0], lr.intercept_))
    print("  MSE: {:.4f}, R2: {:.4f}\n".format(mean_squared_error(y_test, y_pred), r2_score(y_test, y_pred)))

def multivariate_regression(X, y, test_size=0.2, random_state=42):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=test_size, random_state=random_state)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    print("Multivariate Linear Regression (all features):")
    print("  Coefficients:", np.round(lr.coef_, 4))
    print("  Intercept: {:.4f}".format(lr.intercept_))
    print("  MSE: {:.4f}, R2: {:.4f}\n".format(mean_squared_error(y_test, y_pred), r2_score(y_test, y_pred)))

def main():
    X, y = load_diabetes(return_X_y=True)
    # Univariate (use BMI-like feature index 2)
    univariate_regression(X, y, feature_index=2)
    # Multivariate
    multivariate_regression(X, y)

if __name__ == "__main__":
    main()