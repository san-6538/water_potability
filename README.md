

# Water Potability Prediction

This project focuses on predicting the potability of water based on physicochemical properties using machine learning models, ensemble techniques, and data balancing methods.

## Dataset

* Source: [Water Potability Dataset](https://www.kaggle.com/datasets/adityakadiwal/water-potability)
* Features include: `ph`, `Hardness`, `Solids`, `Chloramines`, `Sulfate`, `Conductivity`, `Organic_carbon`, `Trihalomethanes`, `Turbidity`
* Target: `Potability` (0 = Not Potable, 1 = Potable)
* Original class distribution:

  * 0 (Not Potable): 1598 samples
  * 1 (Potable): 1022 samples
  * Imbalance ratio: ~1.56:1

---

## Approaches Implemented

### 1. Stacking Ensemble without Class Balancing

* **Base Models**: LightGBM, XGBoost, Random Forest
* **Meta-Model**: Logistic Regression
* **Train-Test Split**: 80%-20% stratified
* **Metrics:**

| Metric            | Value |
| ----------------- | ----- |
| Accuracy          | 0.61  |
| Precision         | 0.50  |
| Recall            | 0.551 |
| F1 Score          | 0.524 |
| ROC-AUC           | 0.657 |
| Average Precision | 0.599 |

* **Observations:**

  * Model struggles with minority class (`Potable`) due to imbalance.
  * Precision for potable class is low (0.50), indicating many false positives.
  * Recall is slightly better (0.551), but still suboptimal for real-world usage.

---

### 2. Stacking Ensemble with SMOTE Oversampling

* **SMOTE** used to balance training dataset:

  * Positive (Potable) samples increased from 1022 → 1598
  * Negative (Not Potable) samples kept at 1598
* **Models**: Same as above
* **Metrics:**

| Metric            | Value |
| ----------------- | ----- |
| Accuracy          | 0.65  |
| Precision         | 0.555 |
| Recall            | 0.512 |
| F1 Score          | 0.533 |
| ROC-AUC           | 0.668 |
| Average Precision | 0.610 |

* **Observations:**

  * SMOTE balanced the class distribution, allowing the model to better learn patterns for potable water.
  * Accuracy, precision, F1 score, ROC-AUC, and AP improved.
  * Recall slightly decreased due to trade-off with precision.
  * Overall, the model is more reliable for real-world classification of potable vs non-potable water.

---

## Visualization and Explainability

1. **Confusion Matrix**: Shows class-wise prediction performance.
2. **ROC Curve**: Visual evaluation of classification threshold performance.
3. **Precision-Recall Curve**: Helpful for imbalanced classification tasks.
4. **SHAP (SHapley Additive exPlanations)**:

   * Global feature importance (bar chart)
   * Detailed feature impact (beeswarm plot)
   * Local explanations for individual predictions (force plots)

> All plots are saved as PNG files in the project directory for reporting and analysis.

---

## Key Improvements Using SMOTE

* Corrected the **class imbalance**, enabling better learning for minority class (`Potable` water).
* Increased **accuracy** and **weighted F1-score**, giving a more robust overall model.
* Improved **precision and average precision**, reducing false positives in potable predictions.
* Enabled **fairer evaluation of model performance** across classes, reflected in ROC-AUC improvement.

---

## Next Steps

* **Hyperparameter Tuning**: Fine-tune each base model to further improve recall without hurting precision.
* **Alternative Sampling Methods**: Try ADASYN, Borderline-SMOTE, or ensemble-based imbalance handling.
* **Threshold Optimization**: Adjust classification threshold to balance precision vs recall based on real-world requirements.
* **Deployment**: Build a pipeline for predicting water potability for new samples in real-time.

---

## Requirements

```bash
pip install pandas matplotlib seaborn scikit-learn lightgbm xgboost shap imbalanced-learn
```

---

## Authors

**Sachin Kumar** 

Do you want me to do that next?
