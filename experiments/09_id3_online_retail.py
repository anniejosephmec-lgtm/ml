"""Q9: Entropy/information-gain decision tree for Online Retail customers.
Place 'Online Retail.xlsx' beside this project before running.
This is an ID3-style tree using entropy; scikit-learn's tree implementation
supports entropy but uses its own split-search details.
"""
from pathlib import Path
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

path = Path("Online Retail.xlsx")
if not path.exists():
    raise SystemExit("Please download the UCI Online Retail dataset and place 'Online Retail.xlsx' in this folder.")

df = pd.read_excel(path)
df = df.dropna(subset=["CustomerID", "Quantity", "UnitPrice", "InvoiceNo"])
df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
df["TotalSpend"] = df["Quantity"] * df["UnitPrice"]
# Build one row per customer and create a simple high/low spending label.
customer = df.groupby("CustomerID").agg(
    Orders=("InvoiceNo", "nunique"),
    Items=("Quantity", "sum"),
    Spend=("TotalSpend", "sum")
).reset_index(drop=True)
customer["HighSpender"] = (customer["Spend"] >= customer["Spend"].median()).astype(int)
X = customer[["Orders", "Items", "Spend"]]
y = customer["HighSpender"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=42)
model.fit(X_train, y_train)
print("Accuracy:", round(accuracy_score(y_test, model.predict(X_test)), 3))
print("Feature importance:")
for name, score in sorted(zip(X.columns, model.feature_importances_), key=lambda x: -x[1]):
    print(f"{name}: {score:.3f}")
plt.figure(figsize=(12, 6))
plot_tree(model, feature_names=X.columns, class_names=["Low spend", "High spend"],
          filled=True, rounded=True)
plt.tight_layout()
plt.savefig("id3_online_retail_tree.png", dpi=150)
print("Tree saved as id3_online_retail_tree.png")
