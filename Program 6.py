
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import numpy as np

def build_and_run_nn():
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense
    except Exception as e:
        print("TensorFlow/Keras not available. Install tensorflow to run this script.")
        return

    iris = load_iris()
    X, y = iris.data, iris.target

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    ohe = OneHotEncoder(sparse=False)
    y_onehot = ohe.fit_transform(y.reshape(-1,1))

    X_train, X_test, y_train, y_test = train_test_split(Xs, y_onehot, test_size=0.3, random_state=42)

    model = Sequential([
        Dense(16, input_shape=(X_train.shape[1],), activation='relu'),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=60, batch_size=8, verbose=0)
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Neural Network test accuracy: {acc:.4f}")

if __name__ == "__main__":
    build_and_run_nn()