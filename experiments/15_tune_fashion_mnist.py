"""Q15: Small hyperparameter comparison for an MLP on Fashion MNIST."""
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from time import time

print("Downloading Fashion-MNIST from OpenML (first run may take a while)...")
X, y = fetch_openml("Fashion-MNIST", version=1, as_frame=False, return_X_y=True)
X, _, y, _ = train_test_split(X, y, train_size=4000, random_state=42, stratify=y)
X = X / 255.0
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

settings = [
    (0.001, 64, 20),
    (0.01, 64, 20),
    (0.001, 128, 20),
    (0.001, 64, 40),
]
for rate, batch, epochs in settings:
    model = MLPClassifier(hidden_layer_sizes=(64,), learning_rate_init=rate,
                          batch_size=batch, max_iter=epochs, random_state=42,
                          early_stopping=True)
    start = time()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(f"lr={rate}, batch={batch}, epochs={epochs}: "
          f"accuracy={accuracy_score(y_test, pred):.3f}, time={time()-start:.1f}s")
