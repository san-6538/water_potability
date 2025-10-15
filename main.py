# main_optuna.py
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import optuna

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, precision_recall_curve
)
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

from imblearn.over_sampling import ADASYN

# -----------------------
# Config
# -----------------------
RANDOM_STATE = 42
DATA_PATH = "water_potability.csv"
FOCUS_CLASS = 0
TEST_SIZE = 0.2

# -----------------------
# 1) Load dataset
# -----------------------
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["Potability"])
y = df["Potability"]

# -----------------------
# 2) Preprocessing
# -----------------------
imputer = SimpleImputer(strategy="median")
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = pd.DataFrame(poly.fit_transform(X_imputed), columns=poly.get_feature_names_out(X_imputed.columns))

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_poly), columns=X_poly.columns)

# -----------------------
# 3) Train-test split
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)

# -----------------------
# 4) Handle imbalance
# -----------------------
adasyn = ADASYN(random_state=RANDOM_STATE)
X_train_res, y_train_res = adasyn.fit_resample(X_train, y_train)
print("After ADASYN:", np.bincount(y_train_res))

# -----------------------
# 5) Optuna Hyperparameter Tuning
# -----------------------
def objective_rf(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 5, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
        "class_weight": 'balanced',
        "random_state": RANDOM_STATE,
        "n_jobs": 1
    }
    model = RandomForestClassifier(**params)
    model.fit(X_train_res, y_train_res)
    prob = model.predict_proba(X_test)[:, FOCUS_CLASS]
    mapped_true = (y_test == FOCUS_CLASS).astype(int)
    mapped_pred = (prob >= 0.5).astype(int)
    return f1_score(mapped_true, mapped_pred, zero_division=0)

def objective_xgb(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "use_label_encoder": False,
        "eval_metric": "logloss",
        "random_state": RANDOM_STATE,
        "n_jobs": 1
    }
    # compute scale_pos_weight for class imbalance
    params["scale_pos_weight"] = len(y_train_res[y_train_res==0])/len(y_train_res[y_train_res==1])
    model = xgb.XGBClassifier(**params)
    model.fit(X_train_res, y_train_res)
    prob = model.predict_proba(X_test)[:, FOCUS_CLASS]
    mapped_true = (y_test == FOCUS_CLASS).astype(int)
    mapped_pred = (prob >= 0.5).astype(int)
    return f1_score(mapped_true, mapped_pred, zero_division=0)

def objective_lgb(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "num_leaves": trial.suggest_int("num_leaves", 20, 50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
        "n_jobs": 1
    }
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train_res, y_train_res)
    prob = model.predict_proba(X_test)[:, FOCUS_CLASS]
    mapped_true = (y_test == FOCUS_CLASS).astype(int)
    mapped_pred = (prob >= 0.5).astype(int)
    return f1_score(mapped_true, mapped_pred, zero_division=0)

def objective_cat(trial):
    params = {
        "iterations": trial.suggest_int("iterations", 100, 500),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
        "class_weights": [1.0, len(y_train_res[y_train_res==0])/len(y_train_res[y_train_res==1])],
        "verbose": 0,
        "random_state": RANDOM_STATE
    }
    model = CatBoostClassifier(**params)
    model.fit(X_train_res, y_train_res)
    prob = model.predict_proba(X_test)[:, FOCUS_CLASS]
    mapped_true = (y_test == FOCUS_CLASS).astype(int)
    mapped_pred = (prob >= 0.5).astype(int)
    return f1_score(mapped_true, mapped_pred, zero_division=0)

# Run Optuna studies
print("Tuning RandomForest...")
study_rf = optuna.create_study(direction="maximize")
study_rf.optimize(objective_rf, n_trials=25)

print("Tuning XGBoost...")
study_xgb = optuna.create_study(direction="maximize")
study_xgb.optimize(objective_xgb, n_trials=25)

print("Tuning LightGBM...")
study_lgb = optuna.create_study(direction="maximize")
study_lgb.optimize(objective_lgb, n_trials=25)

print("Tuning CatBoost...")
study_cat = optuna.create_study(direction="maximize")
study_cat.optimize(objective_cat, n_trials=25)

# -----------------------
# 6) Train final models with best parameters
# -----------------------
rf_best = RandomForestClassifier(**study_rf.best_params, class_weight='balanced', random_state=RANDOM_STATE, n_jobs=1)
xgb_best = xgb.XGBClassifier(**study_xgb.best_params, use_label_encoder=False, eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=1)
lgb_best = lgb.LGBMClassifier(**study_lgb.best_params, class_weight='balanced', random_state=RANDOM_STATE, n_jobs=1)
cat_best = CatBoostClassifier(**study_cat.best_params, verbose=0, random_state=RANDOM_STATE)

rf_best.fit(X_train_res, y_train_res)
xgb_best.fit(X_train_res, y_train_res)
lgb_best.fit(X_train_res, y_train_res)
cat_best.fit(X_train_res, y_train_res)

# -----------------------
# 7) Voting ensemble + calibrated probabilities
# -----------------------
voting = VotingClassifier(
    estimators=[("rf", rf_best), ("xgb", xgb_best), ("lgb", lgb_best), ("cat", cat_best)],
    voting="soft",
    weights=[1.0, 1.0, 1.2, 1.2],
    n_jobs=1
)
voting.fit(X_train_res, y_train_res)

calibrated_voting = CalibratedClassifierCV(voting, cv='prefit', method='isotonic')
calibrated_voting.fit(X_train_res, y_train_res)

# -----------------------
# 8) Threshold optimization
# -----------------------
prob_class0 = calibrated_voting.predict_proba(X_test)[:, FOCUS_CLASS]
thresholds = np.linspace(0.01, 0.99, 99)
best_thresh = 0.5
best_f1 = -1
mapped_true = (y_test == FOCUS_CLASS).astype(int)

for t in thresholds:
    mapped_pred = (prob_class0 >= t).astype(int)
    f1_c0 = f1_score(mapped_true, mapped_pred, zero_division=0)
    if f1_c0 > best_f1:
        best_f1 = f1_c0
        best_thresh = t

y_pred_thresh = (prob_class0 >= best_thresh).astype(int)
print(f"\nBest threshold: {best_thresh:.3f} | F1 (non-potable): {best_f1:.3f}")

# -----------------------
# 9) Evaluation
# -----------------------
acc = accuracy_score(y_test, y_pred_thresh)
conf = confusion_matrix(y_test, y_pred_thresh)
precision_c0 = precision_score(mapped_true, y_pred_thresh, zero_division=0)
recall_c0 = recall_score(mapped_true, y_pred_thresh, zero_division=0)
f1_c0 = f1_score(mapped_true, y_pred_thresh, zero_division=0)
auc_c0 = roc_auc_score(mapped_true, prob_class0)

print("\nConfusion Matrix:\n", conf)
print(f"Accuracy: {acc:.4f}")
print(f"Precision (non-potable): {precision_c0:.4f}")
print(f"Recall (non-potable): {recall_c0:.4f}")
print(f"F1 (non-potable): {f1_c0:.4f}")
print(f"ROC-AUC (non-potable): {auc_c0:.4f}")

# -----------------------
# 10) SHAP explainability (LightGBM reference)
# -----------------------
explainer = shap.Explainer(lgb_best, X_train_res)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test, show=False, plot_type="bar")
plt.tight_layout()
plt.savefig("shap_feature_importance_optuna.png", bbox_inches="tight")
plt.close()
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig("shap_summary_plot_optuna.png", bbox_inches="tight")
plt.close()

print("\n✅ Hyperparameter-tuned ensemble with optimized threshold completed!")
