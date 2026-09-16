import joblib
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification, make_regression

# 1. Logistic Regression (Classification)
X, y = make_classification(n_samples=100, n_features=4, random_state=42)
clf1 = LogisticRegression()
clf1.fit(X, y)
joblib.dump(clf1, 'logistic_regression_model.pkl')

# 2. Linear Regression (Regression)
X, y = make_regression(n_samples=100, n_features=4, random_state=42)
reg = LinearRegression()
reg.fit(X, y)
joblib.dump(reg, 'linear_regression_model.pkl')

# 3. Decision Tree (Classification)
X, y = make_classification(n_samples=100, n_features=4, random_state=42)
clf2 = DecisionTreeClassifier(random_state=42)
clf2.fit(X, y)
joblib.dump(clf2, 'decision_tree_model.pkl')

print("Created 3 dummy model files: logistic_regression_model.pkl, linear_regression_model.pkl, decision_tree_model.pkl")
