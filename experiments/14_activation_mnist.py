"""Q14: Compare Sigmoid (logistic), ReLU and Tanh activations on MNIST subset."""
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from time import time

print("Downloading MNIST from OpenML (first run may take a while)...")
X, y = fetch_openml("mnist_784", version=1, as_frame=False, return_X_y=True)
X, _, y, _ = train_test_split(X, y, train_size=5000, random_state=42, stratify=y)
X = X / 255.0
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
for activation in ["logistic", "relu", "tanh"]:
    model = MLPClassifier(hidden_layer_sizes=(64,), activation=activation,
                           max_iter=30, random_state=42, early_stopping=True)
    start = time()
    model.fit(X_train, y_train)
    elapsed = time() - start
    pred = model.predict(X_test)
    print(f"{activation}: accuracy={accuracy_score(y_test, pred):.3f}, "
          f"time={elapsed:.1f}s, iterations={model.n_iter_}")
