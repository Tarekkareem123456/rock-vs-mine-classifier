# 🪨 Rock vs Mine Classifier

A machine learning project that classifies sonar signals as Rock or Mine.

---

## 📌 Problem Statement

Sonar signals bounce off objects underwater.
The goal is to build a model that can distinguish between a **Rock** and a **Mine**
based on 60 sonar frequency features.

---

## 📊 Dataset

- **Source:** UCI Machine Learning Repository
- **Samples:** 208
- **Features:** 60 sonar signal frequencies
- **Classes:** Rock (R) = 97 | Mine (M) = 111

---

## 🛠️ Approach

| Step | Description |
|------|-------------|
| 1. EDA | Explored class distribution and signal patterns |
| 2. Feature Engineering | Added 8 statistical features (mean, std, energy, etc.) |
| 3. Preprocessing | StandardScaler + Train/Test Split |
| 4. Feature Selection | Variance Threshold (68 → 40 features) |
| 5. Modeling | Logistic Regression, Random Forest, SVM, Gradient Boosting |
| 6. Tuning | GridSearchCV for hyperparameter optimization |
| 7. Deployment | Streamlit web application |

---

## 🏆 Results

| Metric | Value |
|--------|-------|
| Test Accuracy | 85.7% |
| ROC-AUC | 96.3% |
| Mine Recall | 96% |
| Rock Precision | 93% |

**Best Model:** Gradient Boosting Classifier

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/Tarekkareem123456/rock-vs-mine-classifier.git
cd rock-vs-mine-classifier
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

---

## 🧰 Technologies

| Library | Usage |
|---------|-------|
| Pandas | Data manipulation |
| Scikit-learn | Modeling & evaluation |
| Matplotlib/Seaborn | Visualization |
| Streamlit | Web deployment |
| Joblib | Model saving |