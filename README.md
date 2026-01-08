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
## ▶️ How to Run the Project

Follow the steps below to run the project locally.

---

### 1️⃣ Clone the Repository
```bash
git clone <your-github-repo-link>
cd Eco_Type_Forest_Prediction
```

---

### 2️⃣ Create and Activate Virtual Environment

**Windows**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac / Linux**
```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 3️⃣ Install Required Dependencies
```bash
pip install -r requirements.txt
```

---

### 4️⃣ Run Model Training (One-Time)

This step trains the final model and saves it as `.pkl` files.

```bash
python main.py
```

✔ This will generate:
- `forest_cover_model.pkl`
- `selected_features.pkl`
- `label_encoder.pkl`

---

### 5️⃣ Run the Streamlit Application
```bash
streamlit run app.py
```

The application will open in your browser and allow you to:
- Enter feature values manually
- Predict the forest cover type

---

### 6️⃣ (Optional) View Analysis Notebooks
EDA and model comparison can be viewed using Jupyter Notebook:

```bash
jupyter notebook
```

Open:
- `notebooks/EDA_Forest_Cover.ipynb`
- `notebooks/Model_Comparison.ipynb`

---

## ✅ Notes
- Ensure Python 3.8+ is installed
- Model training is done only once
- Streamlit app uses the saved model for prediction



## 🏁 Conclusion
EcoType demonstrates a complete end-to-end machine learning pipeline—from data analysis and model comparison to deployment—providing a practical solution for forest cover type prediction.

## 👤 Author
Sathishkumar CB
