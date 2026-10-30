"""Q13: MLP on Wine Quality; compare hidden-layer architectures."""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
data = pd.read_csv(url, sep=";")
X, y = data.drop("quality", axis=1), data["quality"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
for layers in [(32,), (64, 32), (64, 32, 16)]:
    model = make_pipeline(StandardScaler(),
        MLPClassifier(hidden_layer_sizes=layers, max_iter=300, random_state=42,
                      early_stopping=True))
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(f"Hidden layers {layers}: accuracy={accuracy_score(y_test, pred):.3f}")
