# ❤️ Heart Disease Risk Predictor

A Machine Learning web application that predicts whether a person is at risk of heart disease based on clinical parameters. The model was developed using Python and Scikit-learn and deployed with Streamlit.

## 🌐 Live Demo

https://ml-heart-risk-predictor.streamlit.app/

---

## 📌 Project Overview

This project follows a complete Machine Learning workflow, including:

- Data Loading
- Exploratory Data Analysis (EDA)
- Data Preprocessing
- Feature Selection
- Model Training
- Model Evaluation
- Model Saving
- Streamlit Deployment

---

## 📂 Dataset

The project uses the Heart Disease dataset containing patient health information such as:

- Age
- Sex
- Chest Pain Type
- Resting Blood Pressure
- Cholesterol
- Fasting Blood Sugar
- Resting ECG
- Maximum Heart Rate
- Exercise Induced Angina
- Oldpeak
- ST Slope
- Number of Major Vessels (ca)
- Thalassemia (thal)

**Target**

- 0 → No Heart Disease
- 1 → Heart Disease

---

## 🔍 Exploratory Data Analysis (EDA)

The notebook includes:

- Dataset exploration
- Missing value analysis
- Duplicate value checking
- Statistical summary
- Correlation analysis
- Feature distribution visualization
- Target class distribution

---

## ⚙️ Data Preprocessing

The following preprocessing steps were performed:

- Handling missing values (if any)
- Feature encoding
- Feature scaling
- Train-Test Split

---

## 🤖 Machine Learning

The model was trained using Scikit-learn and evaluated on the test dataset.

Evaluation includes:

- Accuracy Score
- Confusion Matrix
- Classification Report

The trained model was saved using Pickle for deployment.

---

## 🛠️ Technologies Used

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Pickle

---

## 📁 Project Structure

```
Heart-Disease-Prediction/
│
├── heart.ipynb
├── heart.csv
├── streamlit_app.py
├── model.pkl
├── scaler.pkl
├── requirements.txt
└── README.md
```

---

## ▶️ Run Locally

Clone the repository:

```bash
git clone https://github.com/<your-username>/Machine-Learning.git
```

Go to the project folder:

```bash
cd Heart-Disease-Prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run streamlit_app.py
```

---

## 📸 Application

The Streamlit application allows users to:

- Enter patient health information
- Predict heart disease risk
- View prediction results instantly

---

## 📚 Learning Outcomes

Through this project, I practiced:

- Exploratory Data Analysis (EDA)
- Data Preprocessing
- Feature Engineering
- Classification Models
- Model Evaluation
- Model Serialization
- Streamlit Deployment

---

## 👨‍💻 Author

**Tushar Prajapati**

Machine Learning | Data Science | AI/ML Enthusiast
