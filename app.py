from flask import Flask, render_template, request
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np

app = Flask(__name__)

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

# Load model and vectorizer
model = pickle.load(open("sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
stop_words = set(stopwords.words('english'))

# Preprocess text
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
    return ' '.join(tokens)

# Sentiment words dictionary
positive_words = ['good', 'great', 'excellent', 'amazing', 'outstanding', 'wonderful', 'love', 'best', 'fantastic', 
                 'perfect', 'happy', 'satisfied', 'impressive', 'exceptional', 'superb', 'fun', 'delightful', 'intuitive']
negative_words = ['bad', 'terrible', 'awful', 'poor', 'disappointing', 'worst', 'hate', 'horrible', 'useless', 
                 'frustrated', 'annoying', 'failure', 'dislike', 'problem', 'waste', 'unreliable', 'crash', 'letdown']

@app.route('/', methods=['GET', 'POST'])
def index():
    result = {
        'sentiment': '',
        'confidence': 0,
        'emoji': '',
        'positive_words': [],
        'negative_words': [],
        'score': 50
    }
    
    if request.method == 'POST':
        review = request.form['review'].strip()
        if review:
            # Preprocess and vectorize
            processed_review = preprocess_text(review)
            vec = vectorizer.transform([processed_review])
            
            # Predict sentiment
            pred = model.predict(vec)[0]
            confidence = np.max(model.predict_proba(vec)) * 100
            
            # Calculate sentiment score (0-100)
            score = confidence if pred == 'positive' else (100 - confidence)
            
            # Extract sentiment words
            words = word_tokenize(review.lower())
            positive_found = [w for w in words if w in positive_words]
            negative_found = [w for w in words if w in negative_words]
            
            # Set emoji
            emoji = '😊👍' if pred == 'positive' else '😞👎'
            
            result = {
                'sentiment': pred.capitalize(),
                'confidence': round(confidence, 2),
                'emoji': emoji,
                'positive_words': list(set(positive_found))[:5],
                'negative_words': list(set(negative_found))[:5],
                'score': round(score, 2)
            }
    
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)