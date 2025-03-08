# Supervised-Learning-Classification-
# Systemic Crisis Prediction in Africa

## Project Overview
This project aims to predict the likelihood of a systemic banking crisis in African countries using economic indicators such as inflation, exchange rates, and sovereign debt defaults. A **Random Forest Classifier** was trained on historical crisis data to classify whether a systemic crisis will occur or not.

## Dataset Description
- **Source:** Kaggle
- **Time Span:** 1860 - 2014
- **Countries Covered:** 13 African nations
- **Key Indicators:**
  - Annual Inflation Rate (CPI)
  - Exchange Rate (USD)
  - Domestic & Sovereign Debt Defaults
  - GDP-weighted Default Rate
  - Banking Crisis Status (Binary)
  
## Data Preprocessing
### 1. **Handling Missing and Corrupt Values**
- Checked for missing values; none found.
- Removed any duplicate records (none detected).

### 2. **Outlier Detection & Handling**
- **Boxplots** were used to detect extreme values.
- Applied **Winsorization (5th-95th percentile)** for:
  - `exch_usd` (Exchange Rate in USD)
  - `inflation_annual_cpi` (Inflation Rate)
- Applied **Log Transformation** for:
  - `gdp_weighted_default` (to normalize distribution)

### 3. **Encoding Categorical Variables**
- Dropped redundant column: `country_code`
- **Label Encoding** for `country`
- **Binary Encoding** for `banking_crisis` (crisis = 1, no crisis = 0)

### 4. **Train-Test Split**
- **80% Training** / **20% Testing**
- Stratified split to maintain class distribution.

## Model Selection & Training
- **Initial Model:** Logistic Regression (baseline)
- **Final Model:** Random Forest Classifier
- **Handling Class Imbalance:** Applied **SMOTE (Synthetic Minority Over-sampling Technique)** to balance the dataset.

## Model Evaluation
| Metric          | Score |
|----------------|-------|
| **Accuracy**   | 97.6% |
| **Precision (Class 1)** | 79% |
| **Recall (Class 1)**    | 94% |
| **F1-Score (Class 1)**  | 86% |
| **Confusion Matrix**    |
| [[192, 4], [1, 15]] |

### **Key Findings:**
- The **high recall (94%)** ensures most crisis cases are correctly detected.
- The model has **some false positives**, but SMOTE improved minority class representation.

## Model Saving
The final **Random Forest model** was saved using `joblib` for future use:
```python
import joblib
joblib.dump(rf_smote, 'random_forest_systemic_crisis.pkl')
```

## Next Steps (Optional Improvements)
- **Hyperparameter tuning** (GridSearchCV) to optimize Random Forest parameters.
- **Experimenting with alternative models** (XGBoost, SVM) for better precision.
- **Feature Engineering** (creating interaction terms, additional economic indicators).

## How to Use the Model
1. **Load the model:**
```python
import joblib
rf_model = joblib.load('random_forest_systemic_crisis.pkl')
```
2. **Make predictions:**
```python
y_pred = rf_model.predict(new_data)
```

## Conclusion
This project successfully predicts systemic banking crises in Africa using historical economic data. The **Random Forest model** provides a strong balance between accuracy and recall, ensuring that crises are detected with minimal false negatives.
