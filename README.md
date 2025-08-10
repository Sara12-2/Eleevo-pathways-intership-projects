# 📚 Student Score Prediction

This project builds a regression model to predict students' exam scores based on factors such as study hours, sleep hours, and class participation. It follows the guidelines of Task 1 from the Arch Technologies internship and uses the [Student Performance Factors dataset](https://www.kaggle.com/datasets/lainguyn123/student-performance-factors) from Kaggle.

---

## 🚀 Project Objectives

- Perform data cleaning and visualization
- Train a linear regression model to predict exam scores
- Evaluate model performance using metrics like R² and MSE
- Experiment with polynomial regression and feature selection

---

## 📦 Dataset

**Source**: [Student Performance Factors – Kaggle](https://www.kaggle.com/datasets/lainguyn123/student-performance-factors)  
**Features Used**:
- `Hours_Studied`
- `Sleep_Hours`
- `Participation` (if available)
- `Exam_Score` (target)

---

## 🛠️ Tools & Libraries

- Python
- Pandas
- NumPy
- Matplotlib & Seaborn
- Scikit-learn

---

## 📈 Workflow

### 1. Data Cleaning
- Standardized column names
- Removed missing values
- Selected relevant features

### 2. Exploratory Data Analysis
- Pair plots and correlation heatmaps
- Insights into feature relationships

### 3. Model Training
- Linear Regression using Scikit-learn
- Polynomial Regression for performance comparison

### 4. Evaluation
- R² Score
- Mean Squared Error
- Actual vs Predicted plots

---

## 🔍 Results

| Model Type             | R² Score | Notes                          |
|------------------------|----------|--------------------------------|
| Linear Regression      | ~0.75–0.85 | Strong correlation with study hours |
| Polynomial Regression  | Slightly higher | Captures non-linear patterns |
| Study Hours Only       | Lower     | Less accurate without other features |

---

## 🎯 Bonus Experiments

- Polynomial regression with degree 2
- Feature selection: comparing models with and without `Participation`
- Visualization of predictions and residuals





