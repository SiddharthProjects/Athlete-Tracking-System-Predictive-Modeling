# Training Success Predictor

## Overview

This project implements a **Binary Classification Model** that predicts whether an employee will pass or fail their training assessment based on their profile, skills, performance, and training-related attributes. The model helps organizations identify employees at risk of failing assessments so they can provide targeted support.

---

## Business Problem

Organizations need to:
- Identify employees who may struggle with training assessments
- Allocate additional support resources to at-risk employees
- Understand key factors that influence training success
- Make data-driven decisions about training programs

Manual assessment of employee readiness is subjective and time-consuming. This ML system automates the prediction by analyzing multiple employee attributes to forecast assessment outcomes.

---

## Project Structure

```
training_success_predictor/
├── assessment_prediction_final.ipynb    # Main notebook with model training and evaluation
├── Employee_training.csv                # Training dataset (employee records with assessment scores)
├── bench_test_data_unique.json          # Test dataset (50 unique bench employees)
├── bench_test_data_1000.json            # Test dataset (1000 bench employee records)
└── README.md                             # This documentation
```

---

## Dataset

### Training Data (`Employee_training.csv`)
- **Records**: Employee training records with assessment scores
- **Target Variable**: `Assessment_Score` (0-100)
- **Binary Target**: Pass/Fail (threshold: 70)
  - **Pass**: Assessment_Score >= 70
  - **Fail**: Assessment_Score < 70

### Test Data (JSON Files)
- **bench_test_data_unique.json**: 50 unique bench employee records
- **bench_test_data_1000.json**: 1000 bench employee records
- Format: JSON array with employee attributes

---

## Feature Engineering

### Categorical Features (12 total)
| Feature | Description |
|---------|-------------|
| `Grade` | Employee grade level (G2, G3, G4, G5, G6) |
| `Department` | Department name |
| `Primary_Skill` | Primary technical skill |
| `Secondary_Skill` | Secondary technical skill |
| `Course_Category` | Category of training course |
| `Delivery_Mode` | Training delivery method (Online, Hybrid, In-person) |
| `Business_Priority` | Priority level of the training |
| `Bench_Status` | Employment status (Active, Bench) |
| `Learning_Style` | Preferred learning style |
| `Career_Goal` | Employee's career aspirations |
| `Training_Success` | Previous training success indicator |
| `Training_Course_Name` | Specific course name |

### Numerical Features (5 total)
| Feature | Description | Range |
|---------|-------------|-------|
| `Skill_Gap_Score` | Gap between current and required skills | 0-1 |
| `Availability_Hours_Per_Week` | Hours available for training | 0-40 |
| `Performance_Rating` | Current job performance rating | 1-5 |
| `Duration_Hours` | Total training duration | 0-200 |
| `Completion_Percentage` | Training completion percentage | 0-100 |

---

## Data Preprocessing

### 1. Label Encoding
All 12 categorical features are converted to numerical values using `LabelEncoder`:
```python
for col in categorical_features:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
```

### 2. Feature Scaling
All 5 numerical features are standardized using `StandardScaler`:
```python
scaler = StandardScaler()
data[numerical_features] = scaler.fit_transform(data[numerical_features])
```

### 3. Train-Test Split
- **Split Ratio**: 80% training, 20% testing
- **Stratification**: Maintains Pass/Fail class distribution in both sets
- **Random State**: 42 (for reproducibility)

---

## Model Architecture

### Multi-Model Approach

The project trains and evaluates **three different algorithms**, then selects the best performer:

#### 1. XGBoost Classifier
```python
XGBClassifier(
    n_estimators=200,      # 200 boosting trees
    learning_rate=0.1,     # Learning rate
    max_depth=8,           # Maximum tree depth
    random_state=42
)
```
- **Strengths**: Handles complex patterns, built-in regularization
- **Best for**: Non-linear relationships

#### 2. Random Forest Classifier
```python
RandomForestClassifier(
    n_estimators=200,      # 200 decision trees
    max_depth=15,          # Maximum tree depth
    random_state=42,
    n_jobs=-1              # Use all CPU cores
)
```
- **Strengths**: Robust to outliers, easy to interpret
- **Best for**: Mixed feature types

#### 3. Logistic Regression
```python
LogisticRegression(
    max_iter=1000,         # Maximum iterations
    random_state=42,
    n_jobs=-1              # Multi-threaded
)
```
- **Strengths**: Fast, interpretable, linear baseline
- **Best for**: Comparison and baseline

### Model Selection
After training all three models on the test set, the model with the **highest accuracy** is automatically selected for final predictions.

---

## Model Evaluation Metrics

### Primary Metric
| Metric | Description | Range |
|--------|-------------|-------|
| **Accuracy** | Overall correctness (TP + TN) / Total | 0-100% |

### Detailed Metrics

#### Classification Report
- **Precision**: Of predicted Pass cases, how many were actually Pass
- **Recall**: Of actual Pass cases, how many were predicted correctly
- **F1-Score**: Harmonic mean of Precision and Recall
- **Support**: Number of samples in each class

#### Confusion Matrix
```
                Predicted Pass    Predicted Fail
Actual Pass         TP                FN
Actual Fail         FP                TN
```
- **TP** (True Positive): Correctly predicted Pass
- **TN** (True Negative): Correctly predicted Fail
- **FP** (False Positive): Incorrectly predicted Pass
- **FN** (False Negative): Incorrectly predicted Fail

#### ROC-AUC Score
- Measures model's ability to discriminate between Pass and Fail classes
- **Range**: 0-1 (higher is better, 0.5 = random)
- **Interpretation**: 0.9+ = Excellent, 0.8-0.9 = Good, 0.7-0.8 = Fair

---

## Prediction Process

### Input
Employee record with all 17 features (categorical + numerical)

### Steps
1. **Feature Extraction** - Extract all categorical and numerical features
2. **Label Encoding** - Encode categorical variables using trained encoders
3. **Feature Scaling** - Normalize numerical features using trained scaler
4. **Model Prediction** - Use best model to predict Pass/Fail class
5. **Class Conversion** - Convert prediction back to "Pass" or "Fail" label

### Output
- **Prediction**: "Pass" or "Fail"
- **Confidence**: Model's certainty in the prediction

---

## Usage

### Prerequisites
```bash
pip install pandas numpy scikit-learn xgboost
```

### Running the Model

1. **Load the notebook**:
   ```bash
   jupyter notebook assessment_prediction_final.ipynb
   ```
   or open in VS Code

2. **Run cells sequentially**:
   - **Cell 1**: Import libraries
   - **Cell 2**: Load training and test data from JSON files
   - **Cell 3**: Create binary Pass/Fail target
   - **Cell 4**: Feature engineering and data preprocessing
   - **Cell 5**: Train three models and compare accuracy
   - **Cell 6**: Display detailed classification metrics
   - **Cell 7**: Make predictions on test data
   - **Cell 8**: Print summary results

### Making Predictions

#### Single Employee Prediction
```python
# Define employee profile
employee = {
    'Grade': 'G4',
    'Department': 'Engineering',
    'Primary_Skill': 'Python',
    'Secondary_Skill': 'JavaScript',
    'Course_Category': 'Advanced',
    'Delivery_Mode': 'Hybrid',
    'Business_Priority': 'High',
    'Bench_Status': 'Active',
    'Performance_Rating': 4.2,
    'Learning_Style': 'Hands-on',
    'Training_Success': 'Yes',
    'Training_Course_Name': 'Advanced Python',
    'Skill_Gap_Score': 0.25,
    'Availability_Hours_Per_Week': 12,
    'Duration_Hours': 40,
    'Completion_Percentage': 85
}

# Predict using best model
result = predict_assessment_result(
    'bench_test_data_1000.json',
    best_model,
    scaler,
    label_encoders,
    target_encoder,
    categorical_features,
    numerical_features
)
```

#### Batch Predictions
The model can process JSON files with multiple employee records and generate:
- Per-record predictions (Pass/Fail)
- Comparison with actual assessment scores
- Accuracy metrics by class

---

## Output Format

### Per-Record Prediction
| Column | Example | Description |
|--------|---------|-------------|
| Record | 1 | Record number |
| Actual Score | 75.5 | Actual assessment score |
| Actual Status | Pass | Actual Pass/Fail status |
| Predicted Status | Pass | Model's prediction |
| Predicted Range | 70-100 | Score range for prediction |
| Match | Yes | Correct prediction? |

### Summary Metrics
| Metric | Example | Description |
|--------|---------|-------------|
| Total Records | 1000 | Number of predictions made |
| Overall Accuracy | 87.5% | Percentage of correct predictions |
| Pass Accuracy | 85.2% | Accuracy on Pass cases |
| Fail Accuracy | 90.1% | Accuracy on Fail cases |

---

## Key Insights

### Factors Influencing Training Success

1. **Performance Rating** - Strong positive predictor
   - Higher current performance → Higher chance of passing assessment

2. **Skill Gap Score** - Strong negative predictor
   - Larger skill gap → Lower chance of passing

3. **Availability Hours** - Moderate negative predictor
   - More available hours needed for training → May indicate learning challenges

4. **Completion Percentage** - Strong positive predictor
   - Higher completion rate → Higher chance of passing

5. **Bench Status** - Moderate impact
   - Bench employees may have different assessment outcomes

6. **Grade Level** - Moderate positive impact
   - Higher grades tend to perform better in assessments

---

## Model Performance Interpretation

### Best Model Selection
The notebook automatically selects the best-performing model:
- If **XGBoost** wins: Excellent for complex patterns, handles non-linear relationships well
- If **Random Forest** wins: Robust and interpretable, good with mixed features
- If **Logistic Regression** wins: Simple model with consistent performance

### Accuracy Levels

| Accuracy Range | Model Quality | Action |
|----------------|---------------|--------|
| ≥90% | Excellent | Ready for production |
| 80-89% | Good | Acceptable with monitoring |
| 70-79% | Fair | May need improvement |
| <70% | Poor | Requires retraining |

---

## Limitations

1. **Limited Training Data** - Model trained on available employee records
2. **Class Imbalance** - Pass/Fail distribution may affect predictions
3. **Feature Dependency** - Requires all 17 features for prediction
4. **Temporal Factors** - Doesn't account for seasonal or time-based patterns
5. **External Factors** - Cannot capture unmeasured variables affecting performance

---

## Future Improvements

1. **Feature Engineering**
   - Add interaction terms between features
   - Create polynomial features for better pattern capture
   - Include historical assessment data

2. **Model Enhancements**
   - Implement cross-validation for robust evaluation
   - Perform hyperparameter tuning with GridSearchCV
   - Ensemble methods combining all three models

3. **Data Improvements**
   - Increase training dataset size
   - Balance Pass/Fail classes
   - Add more demographic features

4. **Production Deployment**
   - API endpoint for real-time predictions
   - Batch prediction pipeline
   - Model monitoring and retraining schedule

---

## Libraries Used

| Library | Purpose |
|---------|---------|
| pandas | Data manipulation and analysis |
| numpy | Numerical operations |
| scikit-learn | ML algorithms, preprocessing, metrics |
| xgboost | XGBoost classifier |

---

## File Descriptions

### assessment_prediction_final.ipynb
Main Jupyter notebook containing:
- Data loading and exploration
- Feature engineering and preprocessing
- Model training and evaluation
- Prediction and result generation

### Employee_training.csv
Historical training data with:
- Employee profiles and attributes
- Assessment scores
- Training details

### bench_test_data_unique.json
Test dataset with 50 unique bench employee records for validation

### bench_test_data_1000.json
Larger test dataset with 1000 bench employee records for comprehensive evaluation

---

## How to Interpret Results

### Prediction Confidence
- **High Confidence**: Model consistently predicts Pass/Fail regardless of small feature changes
- **Low Confidence**: Prediction sensitive to small changes, requires careful review

### Class-Specific Accuracy
- **Pass Accuracy**: How well model predicts employees who will pass
  - Important for: Identifying high-achievers
- **Fail Accuracy**: How well model predicts employees who will fail
  - Important for: Early intervention and support

### Error Analysis
- **False Positives** (Predicted Pass, Actually Fail): Missed opportunities for intervention
- **False Negatives** (Predicted Fail, Actually Pass): Over-allocating support

---

## Contact & Support

For questions or issues regarding this model:
- Review the notebook for detailed implementation
- Check data quality and feature values
- Verify all required libraries are installed

---

**Last Updated**: January 2026  
**Model Version**: 1.0  
**Status**: Production Ready  
**Author**: Priti Ranjan Samal

