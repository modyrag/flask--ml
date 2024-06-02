import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pickle

# Download NLTK resources (if not already downloaded)
nltk.download('vader_lexicon')

# Initialize the SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Save the SentimentIntensityAnalyzer object to a pickle file
with open('sentiment_analyzer.pkl', 'wb') as f:
    pickle.dump(sia, f)
