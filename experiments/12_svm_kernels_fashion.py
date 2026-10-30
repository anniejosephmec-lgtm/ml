"""Q12: Compare linear, polynomial and RBF SVM kernels on Fashion MNIST subset."""
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

print("Downloading Fashion-MNIST from OpenML (first run may take a while)...")
X, y = fetch_openml("Fashion-MNIST", version=1, as_frame=False, return_X_y=True)
# Small subset for reasonable runtime.
X, _, y, _ = train_test_split(X, y, train_size=2500, random_state=42, stratify=y)
X = X / 255.0
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
for kernel in ["linear", "poly", "rbf"]:
    model = SVC(kernel=kernel, gamma="scale")
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(f"{kernel} kernel accuracy: {accuracy_score(y_test, pred):.3f}")
