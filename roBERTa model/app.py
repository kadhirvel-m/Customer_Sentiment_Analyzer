from flask import Flask, request, render_template
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

app = Flask(__name__)

# Load Transformer Model
MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

def polarity_scores_roberta(text):
    encoded_text = tokenizer(text, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    sentiments = ['Negative', 'Neutral', 'Positive']
    max_index = scores.argmax()
    overall_sentiment = sentiments[max_index]
    confidence = scores[max_index] * 100  # Convert to percentage

    return {
        'Negative': round(scores[0] * 100, 2),
        'Neutral': round(scores[1] * 100, 2),
        'Positive': round(scores[2] * 100, 2),
        'Overall': f"{overall_sentiment} ({confidence:.2f}%)"
    }

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        text = request.form["text"]
        result = polarity_scores_roberta(text)
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
