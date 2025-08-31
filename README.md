# Titanic Survival Prediction Model

A machine learning project comparing Logistic Regression and Decision Tree algorithms to predict passenger survival on the RMS Titanic based on passenger characteristics and demographics.

## 📊 Dataset Overview

The Titanic dataset contains information about passengers aboard the RMS Titanic, including:
- **Passenger demographics**: Age, gender, passenger class
- **Family information**: Number of siblings/spouses (SibSp), parents/children aboard (Parch)
- **Ticket details**: Fare, port of embarkation
- **Target variable**: Survival (0 = Did not survive, 1 = Survived)

## 🎯 Project Objective

Develop and compare machine learning models to predict whether a passenger survived the Titanic disaster, focusing on Logistic Regression and Decision Tree algorithms with comprehensive preprocessing pipeline.

## 🛠️ Technologies Used

- **Python 3.x**
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computations
- **Scikit-learn** - Machine learning algorithms and preprocessing
- **Matplotlib** - Data visualization and ROC curve plotting

## 📈 Model Performance

| Model | Accuracy | F1-Score | ROC-AUC |
|-------|----------|----------|---------|
| Logistic Regression | [Update with your results]% | [Update with your results] | [Update with your results] |
| Decision Tree | [Update with your results]% | [Update with your results] | [Update with your results] |

*Note: Update the table above with your actual model performance metrics*

## 🔍 Methodology

### Data Preprocessing
- **Feature Selection**: Removed PassengerId, Name, Ticket, and Cabin columns
- **Missing Value Treatment**: 
  - Categorical features: Most frequent value imputation
  - Numerical features: Median imputation
- **Categorical Encoding**: One-hot encoding with drop_first=True for Sex and Embarked
- **Feature Scaling**: StandardScaler applied to numerical features
- **Data Cleaning**: Dropped rows with all missing values

### Features Used
**Categorical Features:**
- Sex (Male/Female)
- Embarked (Port of embarkation: C/Q/S)

**Numerical Features:**
- Pclass (Passenger class: 1, 2, 3)
- Age (Passenger age)
- SibSp (Number of siblings/spouses aboard)
- Parch (Number of parents/children aboard)
- Fare (Ticket fare)

### Model Implementation

#### 1. Logistic Regression
- **Algorithm**: Logistic Regression with max_iter=1000
- **Strengths**: Interpretable coefficients, probabilistic output
- **Use Case**: Baseline linear model for binary classification

#### 2. Decision Tree
- **Algorithm**: Decision Tree Classifier with max_depth=3
- **Hyperparameters**: Limited depth to prevent overfitting
- **Strengths**: Easy to interpret, handles non-linear relationships

### Model Evaluation
- **Train-Test Split**: 80% training, 20% testing (random_state=42)
- **Metrics Calculated**:
  - Accuracy Score
  - F1-Score
  - ROC-AUC Score
  - Confusion Matrix
  - Classification Report
- **ROC Curve Visualization**: Comparative plot for both models

## 📁 Project Structure

```
titanic-survival-prediction/
│
├── data/
│   └── titanic.csv
│
├── notebooks/
│   └── titanic_analysis.ipynb
│
├── src/
│   └── titanic_model.py
│
├── models/
│   ├── logistic_regression_model.pkl
│   └── decision_tree_model.pkl
│
├── visualizations/
│   └── roc_curve_comparison.png
│
├── requirements.txt
├── README.md
└── main.py
```

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.7 or higher
```

### Installation
1. Clone the repository:
```bash
git clone https://github.com/[your-username]/titanic-survival-prediction.git
cd titanic-survival-prediction
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Run the model:
```bash
python main.py
```

## 📊 Key Implementation Details

### Preprocessing Pipeline
The project uses scikit-learn's `ColumnTransformer` and `Pipeline` for robust preprocessing:

```python
# Categorical pipeline: Imputation → One-hot encoding
cat_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(drop="first"))
])

# Numerical pipeline: Imputation → Scaling
num_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
```

### Model Comparison
Both models are evaluated using identical preprocessing and train-test splits to ensure fair comparison. The ROC-AUC curve visualization provides insight into model performance across different threshold values.

## 💡 Key Insights

Based on the analysis, the following patterns emerged:
- Gender played a significant role in survival rates
- Passenger class was strongly correlated with survival
- Age and family size influenced survival probabilities
- The "women and children first" evacuation protocol is evident in the data

## 📈 Visualizations

The project generates:
- **ROC-AUC Comparison Curve**: Visual comparison of both models' performance
- **Confusion Matrix**: True/False positive and negative breakdown
- **Classification Reports**: Detailed precision, recall, and F1-scores

## 🔮 Future Improvements

- [ ] Implement ensemble methods (Random Forest, Gradient Boosting)
- [ ] Advanced feature engineering (family size, title extraction)
- [ ] Hyperparameter optimization using GridSearchCV
- [ ] Cross-validation for more robust performance estimates
- [ ] Feature importance visualization
- [ ] SHAP values for model interpretability

## 🧪 Reproduction

To reproduce the results:
1. Ensure you have the exact dataset file (`titanic.csv`)
2. Run the preprocessing pipeline
3. Train both models with the specified parameters
4. The `random_state=42` ensures reproducible train-test splits

## 📝 Dependencies

Create a `requirements.txt` file with:
```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

## 📧 Contact

[Your Name] - [your.email@example.com]

Project Link: [https://github.com/[your-username]/titanic-survival-prediction](https://github.com/[your-username]/titanic-survival-prediction)

## 🙏 Acknowledgments

- Kaggle for providing the Titanic dataset
- Scikit-learn community for excellent machine learning tools
- The open-source Python ecosystem

---
*This project demonstrates binary classification techniques with proper preprocessing pipelines and model comparison methodologies.*
