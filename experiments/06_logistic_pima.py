"""Q6: Logistic Regression on Pima Indians Diabetes; compare with/without scaling."""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
           "Insulin", "BMI", "DiabetesPedigree", "Age", "Outcome"]
data = pd.read_csv(url, names=columns)
X, y = data.drop("Outcome", axis=1), data["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "Without scaling": make_pipeline(SimpleImputer(strategy="median"),
                                     LogisticRegression(max_iter=2000)),
    "With scaling": make_pipeline(SimpleImputer(strategy="median"),
                                  StandardScaler(),
                                  LogisticRegression(max_iter=2000))
}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print("\\n", name)
    print("Accuracy :", round(accuracy_score(y_test, pred), 3))
    print("Precision:", round(precision_score(y_test, pred, zero_division=0), 3))
    print("Recall   :", round(recall_score(y_test, pred, zero_division=0), 3))
    print("F1-score :", round(f1_score(y_test, pred, zero_division=0), 3))
