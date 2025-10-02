import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import lightgbm as lgb
from imblearn.over_sampling import SMOTE

# -------------------------
# 1. Load dataset
# -------------------------
df = pd.read_csv("water_potability.csv")

# -------------------------
# 2. Features & target
# -------------------------
X = df.drop("Potability", axis=1)
y = df["Potability"]

# -------------------------
# 3. Handle missing values
# -------------------------
imputer = SimpleImputer(strategy="median")
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# -------------------------
# 4. Train-Test Split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------
# 5. SMOTE Oversampling on train set
# -------------------------
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# -------------------------
# 6. Base models
# -------------------------
lgbm = lgb.LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    class_weight="balanced",
    random_state=42,
    n_jobs=1
)

xgb = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    scale_pos_weight=1,  # No need; SMOTE balances data
    random_state=42,
    n_jobs=1,
    use_label_encoder=False,
    eval_metric="logloss"
)

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight="balanced",
    random_state=42,
    n_jobs=1
)

# -------------------------
# 7. Meta-model
# -------------------------
meta_model = LogisticRegression(max_iter=1000, class_weight="balanced")

# -------------------------
# 8. Stacking ensemble
# -------------------------
stack = StackingClassifier(
    estimators=[("lgbm", lgbm), ("xgb", xgb), ("rf", rf)],
    final_estimator=meta_model,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    n_jobs=1
)

# -------------------------
# 9. Train model on resampled data
# -------------------------
stack.fit(X_train_res, y_train_res)

# -------------------------
# 10. Predictions
# -------------------------
y_pred = stack.predict(X_test)
y_proba = stack.predict_proba(X_test)[:, 1]

# -------------------------
# 11. Metrics
# -------------------------
print("📊 Evaluation Metrics:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))
print("Average Precision (AP):", average_precision_score(y_test, y_proba))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -------------------------
# 12. Confusion Matrix
# -------------------------
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
            xticklabels=["Not Potable", "Potable"],
            yticklabels=["Not Potable", "Potable"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix_smote.png", bbox_inches='tight')
plt.close()

# -------------------------
# 13. ROC Curve
# -------------------------
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc_score(y_test, y_proba):.2f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.savefig("roc_curve_smote.png", bbox_inches='tight')
plt.close()

# -------------------------
# 14. Precision-Recall Curve
# -------------------------
prec, rec, _ = precision_recall_curve(y_test, y_proba)
plt.figure(figsize=(6, 4))
plt.plot(rec, prec, label=f"AP = {average_precision_score(y_test, y_proba):.2f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend(loc="upper right")
plt.savefig("precision_recall_curve_smote.png", bbox_inches='tight')
plt.close()

# -------------------------
# 15. SHAP Explainability
# -------------------------
print("\n🔎 SHAP Explainability...")

lgbm.fit(X_train_res, y_train_res)
explainer = shap.TreeExplainer(lgbm)
shap_values = explainer.shap_values(X_test)

# Binary classifier: shap_values is a list
if isinstance(shap_values, list):
    shap_values = shap_values[0]

# Global feature importance (bar chart)
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig("shap_summary_bar_smote.png")
plt.close()

# Beeswarm plot
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig("shap_beeswarm_smote.png")
plt.close()

# Local explanation for first test sample
shap.force_plot(
    explainer.expected_value,
    shap_values[0, :],
    X_test.iloc[0, :],
    matplotlib=True
)
plt.savefig("shap_force_plot_smote.png", bbox_inches='tight')
plt.close()

print("✅ All SMOTE plots saved as PNG files.")

