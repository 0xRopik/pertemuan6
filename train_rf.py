# python train_rf.py
import pandas as pd, joblib, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report, roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix

df = pd.read_csv("processed_kelulusan.csv")
X = df.drop("Lulus", axis=1); y = df["Lulus"]

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

num_cols = X_train.select_dtypes(include="number").columns
pre = ColumnTransformer([("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols)], remainder="drop")

rf = RandomForestClassifier(n_estimators=300, max_features="sqrt", class_weight="balanced", random_state=42)
pipe = Pipeline([("pre", pre), ("clf", rf)])

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
param = {"clf__max_depth": [None, 12, 20, 30], "clf__min_samples_split": [2, 5, 10]}

gs = GridSearchCV(pipe, param_grid=param, cv=skf, scoring="f1_macro", n_jobs=-1, verbose=1)
gs.fit(X_train, y_train)

best_model = gs.best_estimator_
print("Best Params:", gs.best_params_)

y_test_pred = best_model.predict(X_test)
print("F1(test):", f1_score(y_test, y_test_pred, average="macro"))
print(classification_report(y_test, y_test_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))

y_test_proba = best_model.predict_proba(X_test)[:,1]
print("ROC-AUC:", roc_auc_score(y_test, y_test_proba))

fpr, tpr, _ = roc_curve(y_test, y_test_proba)
plt.plot(fpr, tpr); plt.title("ROC Curve"); plt.savefig("roc_test.png")

prec, rec, _ = precision_recall_curve(y_test, y_test_proba)
plt.plot(rec, prec); plt.title("Precision-Recall Curve"); plt.savefig("pr_test.png")

joblib.dump(best_model, "rf_model.pkl")
print("Model disimpan di rf_model.pkl")
