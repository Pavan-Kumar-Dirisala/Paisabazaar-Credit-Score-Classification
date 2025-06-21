# 💳  Credit Score Classification using Machine Learning

A complete Machine Learning project that classifies individuals' **Credit Score** into categories such as `Poor`, `Standard`, or `Good` based on financial and behavioral features. Built using **CatBoost**, **Random Forest**, **KNN**, and **Logistic Regression**, and deployed with a **Streamlit** web app.

---

## 📊 Project Overview

This project aims to predict an individual’s **Credit Score category** based on features like loan type, credit history, payment behavior, income, and other financial indicators. The goal is to help financial institutions or fintech apps understand consumer creditworthiness effectively and with high accuracy.

We experimented with several machine learning models, conducted extensive evaluation and hyperparameter tuning, and finalized **CatBoost (with raw categorical data and hyperparameter tuning)** as the best-performing model with **83.88% accuracy**.

---

## ✅ Features

- 🚀 Multi-model training and evaluation  
- 🔍 Hyperparameter tuning with GridSearchCV  
- 📈 Model explainability using feature importance  
- 🧪 Tested with multiple unseen data examples  
- 🌐 Deployed using **Streamlit**  
- ☁️ Hosted on **Hugging Face Spaces**

---

## 🏆 Final Model Performance

| **Model**                | **Accuracy** | **Key Observations**                                        |
| ------------------------ | ------------ | ----------------------------------------------------------- |
| Logistic Regression       | 64.06%       | Weak precision for class `2`, many misclassifications       |
| Logistic Regression (tuned) | 64.06%    | No improvement due to dataset linearity limitations         |
| Random Forest             | 78.32%       | Strong performance; best for class `1`                      |
| K-Nearest Neighbors       | 78.67%       | Balanced performance; good for classes `0` and `1`          |
| CatBoost (encoded data)   | 74.64%       | Slightly underperformed due to label encoding               |
| CatBoost (raw categorical) | 83.41%      | Great boost with categorical feature handling               |
| **CatBoost (tuned)**      | **83.88%**   | ✅ Final model with highest accuracy after tuning            |

---

## 📁 Project Structure

```
Paisabazaar-Credit-Score-Classification/
│
├──catboost_info/
├── app.py                   
├── best_catboost_model.pkl  
├── requirements.txt         
├── dataset-2.csv                    
├── PaisaBazaar_ML_Submission.ipynb               
└── README.md
```

---

## 🛠️ Setup Instructions

### 🔧 Installation

1. **Clone the Repository**

```bash
git clone https://github.com/your-username/credit-score-classification.git
cd credit-score-classification
```

2. **Install Dependencies**

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Streamlit App

```bash
streamlit run app.py
```

The app will open in your default browser. You can input financial details to predict the credit score category.
---


## 🌍 Live Demo

👉 **Streamlit App**: [Click to open]([https://your-streamlit-url](https://creditworthinessapp.streamlit.app))  
👉 **Hugging Face Space**: [View on Hugging Face](https://huggingface.co/spaces/your-username/credit-score-app)



## 👨‍💻 Author

Made with 💡 by **Pavan Kumar Dirisala**  
📧 [LinkedIn](https://www.linkedin.com/in/pavankumardirisala)  
🔗 [GitHub](https://github.com/pavankumardirisala)

---
