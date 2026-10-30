"""Q10: Compare Logistic Regression and Decision Tree on Adult Income."""
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("Downloading Adult Income from OpenML...")
data = fetch_openml("adult", version=2, as_frame=True)
X, y = data.data, data.target.astype(str).str.strip().str.replace(".", "", regex=False)
num_cols = X.select_dtypes(include=["number"]).columns.tolist()
cat_cols = X.select_dtypes(exclude=["number"]).columns.tolist()
preprocess = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                      ("scale", StandardScaler())]), num_cols),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_cols)
])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
models = {
    "Logistic Regression": Pipeline([("prep", preprocess),
                                     ("model", LogisticRegression(max_iter=1500))]),
    "Decision Tree": Pipeline([("prep", preprocess),
                               ("model", DecisionTreeClassifier(max_depth=8, random_state=42))])
}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print("\\n", name)
    print("Accuracy :", round(accuracy_score(y_test, pred), 3))
    print("Precision:", round(precision_score(y_test, pred, pos_label=">50K", zero_division=0), 3))
    print("Recall   :", round(recall_score(y_test, pred, pos_label=">50K", zero_division=0), 3))
    print("F1-score :", round(f1_score(y_test, pred, pos_label=">50K", zero_division=0), 3))
