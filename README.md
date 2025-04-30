# 🎬 Sentiment Analysis on Movie Reviews

![Sentiment Analysis](https://img.shields.io/badge/NLP-Sentiment%20Analysis-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

A clean and effective NLP-based project that classifies movie reviews as **positive** or **negative** using Python, NLTK, and scikit-learn. Designed for students, developers, and NLP enthusiasts, this project helps interpret sentiments behind textual reviews — simple, fast, and accurate.

🌐 **Live Demo**: [Sentiment Analyzer](https://sentiment-analyzer-5s38.onrender.com)

---

## 🚀 Features
- Predicts sentiment: **Positive** or **Negative**
- Cleaned and preprocessed text pipeline using **NLTK**
- **TF-IDF** vectorization + **Logistic Regression** classification
- Simple and clean Flask-based web interface
- Ideal for beginners in NLP and Machine Learning

---

## 🛠️ Tech Stack
- **Python 3.8+**
- **NLTK** – Tokenization, stopword removal
- **scikit-learn** – TF-IDF & Logistic Regression
- **Flask** – Lightweight backend
- **HTML/CSS** – Minimal front-end for user interaction
- **Render** – App deployment platform

---

## 📋 How It Works
1. **Text Preprocessing**:
   - Tokenization, stopwords removal, lowercasing
2. **Vectorization**:
   - TF-IDF to convert text into numeric features
3. **Model Training**:
   - Logistic Regression for binary classification
4. **Web Interface**:
   - Users can input review → see sentiment instantly

---

## 📦 Installation Guide

### Requirements:
- Python 3.8+
- `pip` for dependency management

### Setup Steps:
```bash
# Step 1: Clone the repository
git clone https://github.com/your-username/sentiment-analyzer.git
cd sentiment-analyzer

# Step 2: (Optional) Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Step 3: Install dependencies
pip install -r requirements.txt

# Step 4: Download NLTK data
python
>>> import nltk
>>> nltk.download('movie_reviews')
>>> nltk.download('punkt')
>>> nltk.download('stopwords')
>>> exit()

# Step 5: Run the app
python app.py
