import gradio as gr
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


with gr.Blocks() as demo:
    name = gr.Textbox(label="Your Text")
    output = gr.Textbox(label="sentiment label")
    greet_btn = gr.Button("check")
    greet_btn.click(fn=analyze_sentiment, inputs=name, outputs=output, api_name="analyze_sentiment")

if __name__ == "__main__":
    demo.launch()
