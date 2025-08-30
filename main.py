import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import lightgbm as lgb
import joblib

# 1. Load data
df = pd.read_csv("water_potability.csv")

# 2. Separate features and target
X = df.drop("Potability", axis=1)
y = df["Potability"]

# 3. Impute missing values using median and keep column names
imputer = SimpleImputer(strategy="median")
X_imputed_array = imputer.fit_transform(X)
X_imputed = pd.DataFrame(X_imputed_array, columns=X.columns)

# 4. Train-test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Set up LightGBM classifier with class imbalance handling
lgb_clf = lgb.LGBMClassifier(
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)

# 6. Hyperparameter tuning with StratifiedKFold CV
param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [5, 10, 15],
    'learning_rate': [0.01, 0.05, 0.1],
    'num_leaves': [20, 31, 40]
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(
    lgb_clf,
    param_grid,
    cv=cv,
    scoring='accuracy',
    verbose=2,
    n_jobs=-1
)

# 7. Fit GridSearch to training data
grid.fit(X_train, y_train)

# 8. Best model from GridSearch
best_model = grid.best_estimator_

# 9. Predict on test data
y_pred = best_model.predict(X_test)

# 10. Evaluation metrics
print("Best Hyperparameters:", grid.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 11. Save model and imputer
joblib.dump(best_model, "lgbm_potability_model.pkl")
joblib.dump(imputer, "potability_imputer.pkl")
print("Model and imputer saved as 'lgbm_potability_model.pkl' and 'potability_imputer.pkl'")
