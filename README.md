# 📩 SMS Spam Detection Classifier

This project is a machine learning-based SMS Spam Classifier that uses natural language processing (NLP) techniques to detect whether an incoming message is **spam** or **ham (not spam)**.

---

## 📊 Project Overview

- Built using a labeled dataset of SMS messages.
- Includes full pipeline: text preprocessing → feature extraction → model training → prediction.
- A Flask-based web app is included for local testing.
- This project was developed as part of my portfolio for a **Data Analyst interview**.

---

## 🛠️ Tech Stack

- **Languages**: Python
- **Libraries**: Pandas, Scikit-learn, NLTK, Flask
- **ML Model**: Multinomial Naive Bayes (with TF-IDF vectorization)
- **Deployment**: Flask (Local)

---

## 🔄 Workflow

1. **Data Cleaning & EDA**:
   - Removed punctuation, stopwords, and performed tokenization.
   - Visualized spam vs. ham distribution and message length.

2. **Text Preprocessing**:
   - Used `NLTK` for stopword removal and lemmatization.
   - Vectorized text using `TF-IDF`.

3. **Model Training**:
   - Trained and evaluated using Multinomial Naive Bayes.
   - Achieved high accuracy with minimal overfitting.

4. **Deployment**:
   - Built a simple Flask app (`app.py`) to test live predictions.
   - Model and vectorizer saved using Pickle (`model.pkl`, `vectorizer.pkl`).

---

## 🚀 Getting Started

### 🔧 Requirements
Install the required packages:
```bash
pip install -r requirements.txt

📬 Contact
Dolagovinda Sahu
📧 sahulucky990@gmail.com
🔗 LinkedIn
🔗 GitHub
