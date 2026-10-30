"""Q11: Linear SVM; Setosa vs non-Setosa; plot boundary and margins."""
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

iris = load_iris()
X = iris.data[:, :2]  # two features make a 2D plot
y = (iris.target != 0).astype(int)  # 0 = Setosa, 1 = Non-Setosa
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
model = make_pipeline(StandardScaler(), SVC(kernel="linear"))
model.fit(X_train, y_train)
print("Accuracy:", round(accuracy_score(y_test, model.predict(X_test)), 3))

scaler = model.named_steps["standardscaler"]
svm = model.named_steps["svc"]
X_scaled = scaler.transform(X)
x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
xx, yy = __import__("numpy").meshgrid(
    __import__("numpy").linspace(x_min, x_max, 300),
    __import__("numpy").linspace(y_min, y_max, 300)
)
Z = svm.predict(__import__("numpy").c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.2)
plt.contour(xx, yy, svm.decision_function(__import__("numpy").c_[xx.ravel(), yy.ravel()]).reshape(xx.shape),
            levels=[-1, 0, 1], linestyles=["--", "-", "--"])
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, edgecolors="black")
plt.xlabel("Sepal length (scaled)")
plt.ylabel("Sepal width (scaled)")
plt.title("Linear SVM: Setosa vs Non-Setosa")
plt.tight_layout()
plt.savefig("linear_svm_iris.png", dpi=150)
print("Plot saved as linear_svm_iris.png")
