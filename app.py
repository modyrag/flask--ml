from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle

# Load the SentimentIntensityAnalyzer object from the pickle file
with open('sentiment_analyzer.pkl', 'rb') as f:
    sia = pickle.load(f)

# Define the sentiment analysis function
def analyze_sentiment(text):
    # Perform sentiment analysis
    sentiment_scores = sia.polarity_scores(text)

    # Determine sentiment label based on compound score
    if sentiment_scores['compound'] > 0.05:
        return "Positive"
    elif sentiment_scores['compound'] < -0.05:
        return "Negative"
    else:
        return "Neutral"

# Create a Flask application
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Define a route for the sentiment analysis API
@app.route('/analyze_sentiment', methods=['POST'])
def analyze_sentiment_route():
    if not request.json or 'text' not in request.json:
        return jsonify({"error": "Invalid input"}), 400

    text = request.json['text']
    sentiment_label = analyze_sentiment(text)
    return jsonify({"sentiment": sentiment_label})

if __name__ == "__main__":
    app.run(debug=True)
