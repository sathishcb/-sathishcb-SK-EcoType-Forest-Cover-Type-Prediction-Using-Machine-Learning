# 🌲 EcoType: Forest Cover Type Prediction Using Machine Learning

## 📌 Project Overview
EcoType is a machine learning classification project that predicts the forest cover type of a geographical area using cartographic and environmental features such as elevation, slope, soil type, and distance measures. The project supports environmental monitoring, forestry management, and land-use planning by providing an automated and reliable prediction system.

## 🎯 Problem Statement
To develop a machine learning classification model that accurately predicts the forest cover type based on cartographic variables, enabling efficient forest resource management and ecological analysis.

## 🌿 Domain
Environmental Data & Geospatial Predictive Modeling

## 📚 Skills & Technologies Used
- Exploratory Data Analysis (EDA)
- Data Cleaning & Preprocessing
- Skewness Detection & Handling
- Feature Engineering
- Class Imbalance Handling (SMOTE)
- Classification Models
- Model Evaluation
- Hyperparameter Tuning
- Streamlit Application Development
- Model Deployment

Libraries & Tools:
Python, Pandas, NumPy, Scikit-learn, XGBoost, Imbalanced-learn, Matplotlib, Seaborn, Streamlit, Joblib

## 📊 Dataset Information
- Source: Forest Cover Type Dataset
- Size: 145,891 rows × 13 columns
- Target Variable: Cover_Type (7 classes)

## 🔍 Exploratory Data Analysis (EDA)
EDA was performed in a separate Jupyter notebook to understand feature distributions, skewness, class imbalance, correlations, and feature importance.

Notebook:
- notebooks/EDA_Forest_Cover.ipynb

## ⚙️ Data Preprocessing
- Verified no missing values
- Detected skewed features using skewness metrics
- Applied transformations where required
- Encoded target variable
- Ensured consistent feature selection

## ⚖️ Class Imbalance Handling
SMOTE (Synthetic Minority Oversampling Technique) was applied on the training dataset to balance class distribution.

## 🧠 Model Building & Evaluation

Models trained:
- Logistic Regression
- Decision Tree
- K-Nearest Neighbors (KNN)
- Random Forest
- XGBoost

Evaluation metrics:
- Accuracy
- Confusion Matrix
- Classification Report

### 📈 Model Comparison Summary

| Model | Accuracy |
|------|----------|
| Logistic Regression | 0.72 |
| Decision Tree | 0.97 |
| KNN | 0.95 |
| Random Forest | 0.99 |
| XGBoost | 0.99 |

Best Model Selected: Random Forest

Notebook:
- notebooks/Model_Comparison.ipynb

## 🔧 Hyperparameter Tuning
RandomizedSearchCV was applied to the Random Forest model to optimize performance while keeping training time reasonable.

## 💾 Model Saving
Saved artifacts using joblib:
- forest_cover_model.pkl
- selected_features.pkl
- label_encoder.pkl

## 🌐 Streamlit Application
A Streamlit web application was developed for single-instance prediction using manual numeric inputs.

Run the app:
streamlit run app.py
			
## 📁 Project Structure

```text
Eco_Type_Forest_Prediction/
│
├── data/
│   └── covtype.csv
│
├── notebooks/
│   ├── EDA_Forest_Cover.ipynb
│   └── Model_Comparison.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── skewness_handler.py
│   ├── imbalance_handler.py
│   ├── feature_selection.py
│   └── model_training.py
│
├── main.py
├── app.py
├── forest_cover_model.pkl
├── selected_features.pkl
├── label_encoder.pkl
├── requirements.txt
└── README.md
```

## 🏁 Conclusion
EcoType demonstrates a complete end-to-end machine learning pipeline—from data analysis and model comparison to deployment—providing a practical solution for forest cover type prediction.

## 👤 Author
Sathishkumar CB
