
# 🌍 Natural Disaster Prediction using ML

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Scikit--Learn-F7931A?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-learn">
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas">
  <img src="https://img.shields.io/badge/Numpy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/Matplotlib-007ACC?style=for-the-badge&logo=matplotlib&logoColor=white" alt="Matplotlib">
  <img src="https://img.shields.io/badge/Seaborn-2B2D42?style=for-the-badge&logo=seaborn&logoColor=white" alt="Seaborn">
</p>

<p align="center">
  <b>Predicting natural disasters using machine learning algorithms.</b><br>
  <i>From data preprocessing to ensemble learning, this project provides an end-to-end pipeline for disaster prediction.</i>
</p>

---

## 🧾 Project Overview

This repository focuses on building machine learning models to predict various natural disasters by analyzing historical data. The pipeline includes steps from exploratory data analysis (EDA) and feature engineering to advanced ensemble models and deployment-ready solutions.

---

## 🗃️ Repository Structure

```
Natural-Disaster-Prediction-Using-Machine-Learning/
│
├── Datasets/
│   ├── natural_disasters_dataset.csv
│   ├── preprocessed_data.csv
│   └── new_unseen_dataset.csv
│
├── Code/
│   ├── Complete_code.ipynb
│   ├── Complete_code.py
│   ├── Testing_the_saved_model.ipynb
│   └── Testing_the_saved_model.py
│
├── Reports/
│   ├── Project_Report.pdf
│   ├── Industry_CaseStudy.pdf
│   └── Significant_paper_slides.pdf
│
├── Models/
│   └── [Joblib Saved Model - Link](https://drive.google.com/drive/u/1/folders/1ND3XnuUrvSIkmtFcO-zI6ovv2ykQyWp3)
│
└── README.md
```

---

## 🔍 Key Components & Techniques

| **Phase** | **Details** |
|----------|-------------|
| **1. Data Collection** | Acquired disaster datasets from Kaggle containing timestamps, location, and event types. |
| **2. EDA** | Performed detailed exploratory data analysis with visualizations to understand distribution and correlations. |
| **3. Preprocessing** | Handled missing data, encoded categorical variables, and created the `preprocessed_data.csv`. |
| **4. Feature Selection** | Selected relevant features using Mutual Information and domain knowledge. |
| **5. Model Building** | Implemented Random Forest, SVM, K-NN, and Naive Bayes classifiers. |
| **6. Evaluation** | Used Accuracy, Precision, Recall, and F1-Score to compare models. |
| **7. Data Balancing** | Applied sampling techniques and built ensemble classifiers (Soft/Hard Voting). |
| **8. Hyperparameter Tuning** | Optimized model parameters using GridSearchCV. |
| **9. Deployment** | Saved model using Joblib and tested it on unseen data. |

---

## 🚀 Getting Started

### 📦 Requirements

- Python 3.8+
- Libraries: `numpy`, `pandas`, `scikit-learn`, `seaborn`, `matplotlib`, `joblib`

### 🔧 Setup Instructions

```bash
git clone https://github.com/your-username/Natural-Disaster-Prediction-Using-Machine-Learning.git
cd Natural-Disaster-Prediction-Using-Machine-Learning

# (Optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt  # or install manually
```

### 🧪 Running the Model

```bash
cd Code/
python Complete_code.py

# or test the deployed model
python Testing_the_saved_model.py
```

---

## 📊 Model Performance Snapshot

| **Model**     | **Accuracy** | **F1 Score** | **Precision** | **Recall** |
|---------------|--------------|--------------|----------------|-------------|
| Random Forest | ✅ High       | ✅ High       | ✅ High         | ✅ High      |
| SVM           | Moderate     | Moderate     | Moderate        | Moderate     |
| K-NN          | Variable     | Moderate     | Moderate        | Low          |
| Naive Bayes   | Lower        | Lower        | Moderate        | Moderate     |

> Ensemble techniques improved performance across all metrics.

---

## 👨‍💻 Author

Developed by:

**B. Chaitanya**  
*Data Science & Machine Learning Enthusiast*  
GitHub: [bchaitanya92](https://github.com/bchaitanya92)  
LinkedIn: [BOURISETTI CHAITANYA](https://www.linkedin.com/in/b-chaitanya)

---

## 📄 License

This project is licensed under the MIT License. Feel free to use, modify, and share with attribution.

---

🎉 *Explore. Predict. Protect. Together, we build a safer tomorrow with ML.*
