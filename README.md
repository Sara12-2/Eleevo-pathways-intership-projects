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



# 🛍️ Mall Customer Segmentation

## 🎯 Objective
Segment mall customers into distinct groups based on their **Annual Income** and **Spending Score** using clustering techniques. This helps businesses understand customer behavior and tailor marketing strategies.

---

## 📊 Dataset
- **Source**: [Kaggle - Mall Customers Dataset](https://www.kaggle.com/datasets/kryusufkaya/mall-customers-dataset)
- **Features Used**:
  - `Annual Income (k$)`
  - `Spending Score (1-100)`

---

## 🛠️ Tools & Libraries
- Python 🐍
- Pandas
- Matplotlib & Seaborn
- Scikit-learn

---

## 📌 Project Workflow

### 1. Data Loading & Exploration
- Loaded CSV using Pandas
- Explored basic statistics and feature distributions

### 2. Feature Selection
- Selected `Annual Income` and `Spending Score` for clustering

### 3. Data Scaling
- Applied `StandardScaler` to normalize features

### 4. Visual Exploration
- Used scatter plots to visualize customer distribution

### 5. Optimal Cluster Detection
- Applied **Elbow Method** to find the best number of clusters

### 6. K-Means Clustering
- Clustered customers into 5 segments
- Extracted cluster labels and centroids

### 7. Cluster Visualization
- Plotted clusters in 2D space with distinct colors
- Highlighted centroids using yellow markers

---

## 🌟 Bonus Tasks

### 🔍 DBSCAN Clustering
- Applied **DBSCAN** for density-based clustering
- Detected noise points and non-spherical clusters
- Evaluated clustering using **Silhouette Score**

### 📊 Spending Score Analysis
- Calculated **average spending score per cluster**
- Visualized spending behavior using bar plots
- Provided insights into which clusters represent high-value customers


# 🏦 Loan Approval Prediction (with SMOTE)

This project builds a machine learning pipeline to predict loan approval decisions based on applicant data. It addresses class imbalance using SMOTE and compares the performance of Logistic Regression and Decision Tree classifiers.

---

## 📌 Problem Statement

Financial institutions need reliable systems to assess loan applications. This project aims to predict whether a loan should be **approved (`1`)** or **rejected (`0`)** using features such as income, credit history, and education.

---

## 📦 Dataset

- **Source**: Simulated or anonymized loan application data  
- **Target Variable**: `Loan_Status` (0 = Rejected, 1 = Approved)  
- **Features Used**:
  - `ApplicantIncome`
  - `LoanAmount`
  - `Credit_History`
  - `Education`
  - `Gender`
  - `Married`
  - *(Additional features if available)*

---

## 🛠️ Tools & Libraries

- Python 🐍  
- Pandas, NumPy  
- Scikit-learn  
- imbalanced-learn (SMOTE)  
- Matplotlib, Seaborn  
- ipywidgets (for interactive GUI)

---

## 📈 Workflow

### 1. Data Preprocessing
- Handled missing values
- Encoded categorical variables
- Scaled numerical features
- Split into training and test sets

### 2. Modeling
- Trained two classifiers:
  - Logistic Regression
  - Decision Tree
- Applied **SMOTE** to balance training data

### 3. Evaluation
- Confusion Matrix
- Precision, Recall, F1-score
- Accuracy and class-wise metrics

### 4. Bonus: Interactive GUI
- Built a simple loan prediction interface using `ipywidgets`
- Users can input features and get real-time predictions

---

## 📊 Results

| Model                  | Accuracy | F1-Score (Approved) |
|-----------------------|----------|---------------------|
| Logistic Regression   | 91.3%    | 0.931               |
| Decision Tree         | 98.4%    | 0.987               |
| Logistic + SMOTE      | 92.3%    | 0.937               |
| Decision Tree + SMOTE | 98.0%    | 0.984               |

- **SMOTE** improved recall for minority class (rejected loans)
- **Decision Tree** showed highest accuracy but may overfit
- **Logistic Regression** offers interpretability and solid performance

---

## 🎯 Bonus Experiments

- Compared models with and without SMOTE
- Visualized confusion matrices and classification reports
- Built an interactive predictor using `ipywidgets`
- Explored feature importance and decision boundaries

---

## 📁 Files

- `loan_data.csv` – Raw dataset  
- `loan_model.ipynb` – Full notebook with pipeline  
- `loan_predictor_gui.ipynb` – Interactive GUI using ipywidgets  
- `README.md` – Project documentation

---

## 🚀 Future Enhancements

- Hyperparameter tuning with GridSearchCV  
- Add more features (e.g., employment type, dependents)  
- Deploy as a web app using Streamlit or Flask  
- Integrate explainability tools like SHAP or LIME

---



