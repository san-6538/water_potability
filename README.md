
# 💧 Water Potability Prediction

![Python](https://img.shields.io/badge/Python-3.11-blue) ![Scikit-learn](https://img.shields.io/badge/Scikit--Learn-1.2-green) ![LightGBM](https://img.shields.io/badge/LightGBM-3.3-orange) ![XGBoost](https://img.shields.io/badge/XGBoost-1.7-red) ![Optuna](https://img.shields.io/badge/Optuna-3.1-purple)

Predicting water potability using physicochemical properties with **machine learning**, **ensemble techniques**, **class balancing**, and **explainable AI**.

---

## 📊 Dataset

* **Source:** [Kaggle - Water Potability](https://www.kaggle.com/datasets/adityakadiwal/water-potability)
* **Features:** `ph`, `Hardness`, `Solids`, `Chloramines`, `Sulfate`, `Conductivity`, `Organic_carbon`, `Trihalomethanes`, `Turbidity`
* **Target:** `Potability` (0 = Not Potable, 1 = Potable)
* **Original Class Distribution:**

| Class           | Count |
| --------------- | ----- |
| Not Potable (0) | 1598  |
| Potable (1)     | 1022  |

> Imbalance ratio ~1.56:1

---

## 🏗 Project Pipeline & Improvements

### 1️⃣ Baseline Stacking Ensemble (No Balancing)

* **Models:** LightGBM, XGBoost, Random Forest
* **Meta-Model:** Logistic Regression
* **Metrics:**

| Metric    | Value |
| --------- | ----- |
| Accuracy  | 0.61  |
| Precision | 0.50  |
| Recall    | 0.551 |
| F1 Score  | 0.524 |
| ROC-AUC   | 0.657 |

**Observations:**

* Struggled with minority class (`Potable`) → low precision.
* Recall moderate → risky for real-world deployment.

---

### 2️⃣ SMOTE Oversampling + Ensemble

* Balanced training set with SMOTE → 1598 samples each class
* **Metrics:**

| Metric    | Value |
| --------- | ----- |
| Accuracy  | 0.65  |
| Precision | 0.555 |
| Recall    | 0.512 |
| F1 Score  | 0.533 |
| ROC-AUC   | 0.668 |

**Observations:**

* Balanced learning improved model performance
* Precision and F1 improved, recall slightly decreased

---

### 3️⃣ Hyperparameter-Tuned Ensemble + Threshold Optimization

* Optimized **LightGBM, XGBoost, Random Forest** using Optuna
* **SMOTE + Tomek** applied for class balancing
* **Threshold optimized** for non-potable class → high recall priority

**Performance (focus: non-potable):**

| Metric    | Value  |
| --------- | ------ |
| Accuracy  | 0.3369 |
| Precision | 0.6607 |
| Recall    | 0.9200 |
| F1 Score  | 0.7691 |
| ROC-AUC   | 0.6359 |

**Highlights:**

* Maximized **detection of non-potable water**
* Trade-off: overall accuracy drops → safety-critical focus
* F1 improved for minority class, reducing false negatives

---

---

## 🔧 Requirements

```bash
pip install pandas matplotlib seaborn scikit-learn lightgbm xgboost shap imbalanced-learn optuna
```

---

## 🚀 Improvement Timeline

| Stage                  | Techniques                             | Key Metrics / Learnings                                   |
| ---------------------- | -------------------------------------- | --------------------------------------------------------- |
| Baseline               | Stacking Ensemble                      | Low precision/recall for potable water                    |
| SMOTE                  | Balanced Classes                       | Improved F1, ROC-AUC, precision                           |
| Hyperparam + Threshold | Optimized ensemble + non-potable focus | High recall (0.92), improved F1 (0.769), safe predictions |

---

## 📌 Key Learnings

* Handling **class imbalance** is crucial for real-world applications
* **Ensemble + SMOTE/Tomek + Hyperparameter tuning** enhances model performance
* **Threshold optimization** allows prioritization of critical classes
* **SHAP explainability** increases trust in predictions

---

## 🛠 Next Steps

* Test **ADASYN** or **Borderline-SMOTE** for advanced balancing
* Implement **probability calibration** for more robust thresholds
* Deploy **real-time prediction pipeline**
* Explore **meta-learning ensembles** for further improvement

---

## 👤 Author

**Sachin Kumar** – Machine Learning Enthusiast | AI for Social Good


