"""Q8: KNN on a small Fashion MNIST subset; try different K values."""
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
import numpy as np

print("Downloading Fashion-MNIST from OpenML (first run may take a while)...")
X, y = fetch_openml("Fashion-MNIST", version=1, as_frame=False, return_X_y=True)
# A small, stratified subset makes the lab experiment quicker.
X, _, y, _ = train_test_split(X, y, train_size=6000, random_state=42, stratify=y)
X = X / 255.0
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
for k in [3, 5, 7]:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(f"K={k}: accuracy={accuracy_score(y_test, pred):.3f}")
