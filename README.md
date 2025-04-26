# Advanced Diabetes Risk Prediction System

## Overview
This project implements a sophisticated machine learning system for predicting diabetes risk using an ensemble of models and deep learning. The system achieved 98.75% accuracy and 0.9688 AUC-ROC score on the test set.

## Key Results
```
Final Model Performance:
- Accuracy: 98.75%
- AUC-ROC: 0.9688

Classification Report:
              precision    recall  f1-score   support
           0       0.91      0.95      0.93       168
           1       1.00      0.99      0.99      1832
```

## Technical Components

### 1. Data Preprocessing
- **Feature Engineering:**
  - Created interaction features (e.g., BMI_Age, BMI_Waist_Ratio)
  - Generated polynomial features for key metrics
  - Applied log transformations for skewed distributions
  - Implemented frequency encoding for categorical variables

- **Missing Value Handling:**
  - Categorical: Mode imputation
  - Numerical: Median + random noise imputation

### 2. Target Variable Creation
Diabetes risk is determined by multiple criteria:
- HbA1c ≥ 6.5%
- Fasting Blood Glucose ≥ 126 mg/dL
- Combined risk factors (BMI, Age, Blood Glucose)
- Genetic predisposition consideration

### 3. Model Architecture
The system uses a stacked ensemble approach:

**Base Models:**
1. Random Forest Classifier
   - 200 estimators
   - Max depth: 20
   - Balanced class weights

2. Gradient Boosting Classifier
   - 200 estimators
   - Learning rate: 0.1
   - Max depth: 5

3. Support Vector Machine
   - RBF kernel
   - C=10
   - Balanced class weights

**Deep Learning Model:**
- 4 Dense layers (256→128→64→32 units)
- Batch Normalization
- Dropout (0.4→0.4→0.3→0.2)
- L1/L2 regularization
- Adam optimizer (lr=0.001)

### 4. Class Imbalance Handling
- SMOTETomek for balanced training data
- Stratified sampling for train-test split
- Class weights in base models

### 5. Performance Metrics
The model shows excellent performance:
- **Low-Risk Cases (0):**
  - Precision: 91%
  - Recall: 95%
  - F1-score: 93%

- **High-Risk Cases (1):**
  - Precision: 100%
  - Recall: 99%
  - F1-score: 99%

## Requirements
```python
numpy>=1.19.2
pandas>=1.2.4
scikit-learn>=0.24.2
tensorflow>=2.5.0
imbalanced-learn>=0.8.0
seaborn>=0.11.1
matplotlib>=3.3.4
```

## Usage
```bash
python diabetes_prediction.py
```

## Model Artifacts
The system saves two model files:
- `stack_ensemble_model.pkl`: Stacked ensemble model
- `deep_learning_model.h5`: Neural network model

## Notes
- The high accuracy (98.75%) suggests excellent predictive performance
- Strong performance on both classes indicates good balance despite class imbalance
- The model is particularly effective at identifying high-risk cases (100% precision)

## Future Improvements
1. Feature importance analysis
2. Cross-validation for model stability
3. Hyperparameter optimization
4. External validation dataset testing

## License
MIT License
