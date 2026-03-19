# 💳 Credit Card Fraud Detection

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.2+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.20+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

An end-to-end Machine Learning project to detect fraudulent credit card transactions using multiple ML models, class imbalance handling, and an interactive Streamlit web app.

---

## 🎯 Project Overview

Credit card fraud is a major financial problem worldwide. This project builds a complete fraud detection pipeline that:
- Trains and compares **3 ML models**
- Handles **highly imbalanced data** (only ~2% fraud cases)
- Provides **model explainability** via Feature Importance
- Deploys an **interactive web app** for real-time predictions

---

## 📊 Model Results

| Model | AUC-ROC | PR-AUC | F1 Score | Precision | Recall |
|---|---|---|---|---|---|
| **Logistic Regression** | 1.0000 | 0.9997 | 0.9677 | 0.9375 | 1.0000 |
| **Gradient Boosting** | 0.9999 | 0.9971 | 0.9661 | 0.9828 | 0.9500 |
| **Random Forest** | 0.9999 | 0.9967 | 0.9573 | 0.9825 | 0.9333 |

---

## 🏗️ Project Structure

```
credit-card-fraud-detection/
├── train.py              # Full ML pipeline
├── app.py                # Streamlit web app
├── requirements.txt      # Dependencies
├── data/                 # Dataset folder
│   └── creditcard.csv    # Kaggle dataset (place here)
├── models/               # Saved trained models
│   ├── Gradient_Boosting.pkl
│   ├── Random_Forest.pkl
│   └── Logistic_Regression.pkl
└── outputs/              # Generated plots & results
    ├── eda.png
    ├── roc_pr_curves.png
    ├── confusion_matrices.png
    ├── model_comparison.png
    ├── feature_importance.png
    └── results.csv
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/dimpalshegekar/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Download from [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place `creditcard.csv` in the `data/` folder.

> **No dataset?** The pipeline auto-generates synthetic data for demo purposes.

### 4. Train the models
```bash
python train.py
```

### 5. Launch the web app
```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## 🤖 Models Used

### Logistic Regression
- Simple, interpretable baseline model
- Fast training and prediction
- Uses `class_weight='balanced'` for imbalance

### Random Forest
- Ensemble of decision trees
- Robust to outliers and noise
- Provides feature importance scores

### Gradient Boosting
- Boosted ensemble model
- Best overall performance on tabular data
- Handles non-linear relationships well

---

## ⚖️ Handling Class Imbalance

The dataset has only ~2% fraud cases. Without handling this, models predict "not fraud" every time and achieve 98% accuracy — but catch zero frauds!

**Solution:** `class_weight='balanced'` — automatically adjusts weights inversely proportional to class frequencies.

---

## 📏 Evaluation Metrics

| Metric | Why It Matters |
|---|---|
| **AUC-ROC** | Measures overall discrimination ability |
| **PR-AUC** | Best metric for imbalanced datasets |
| **F1 Score** | Balance between precision and recall |
| **Recall** | How many frauds we actually catch |
| **Precision** | How many flagged transactions are real fraud |

---

## 🛠️ Tech Stack

| Purpose | Library |
|---|---|
| Data manipulation | pandas, numpy |
| ML Models | scikit-learn |
| Visualization | matplotlib, seaborn |
| Web App | Streamlit |
| Model Saving | joblib |

---

## 💡 Key Learnings

- Why **PR-AUC is better than AUC-ROC** for imbalanced fraud detection
- How **class weighting** solves the imbalance problem
- Why **Recall matters more than Precision** in fraud detection
- How **Gradient Boosting** outperforms simpler models on tabular data

---

## 🎯 Interview Talking Points

- Why did you choose PR-AUC over accuracy?
- What's the cost of false negatives vs false positives in fraud?
- How does class weighting differ from SMOTE?
- Why is Gradient Boosting better than Random Forest here?

---

## 👩‍💻 Author

**Dimpal Shegekar**
- GitHub: [@dimpalshegekar](https://github.com/dimpalshegekar)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
