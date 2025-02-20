from flask import Flask, render_template, request
import emoji
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

app = Flask(__name__)

# Load Transformer Model
MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

def polarity_scores_roberta(text):
    text = emoji.demojize(text)  # Convert emojis to text
    encoded_text = tokenizer(text, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    positive = round(scores[2] * 100, 2)
    neutral = round(scores[1] * 100, 2)
    negative = round(scores[0] * 100, 2)

    # Determine overall sentiment label
    if positive > neutral and positive > negative:
        overall = f"😀 Positive ({positive}%)"
    elif neutral > positive and neutral > negative:
        overall = f"😐 Neutral ({neutral}%)"
    else:
        overall = f"😡 Negative ({negative}%)"

    return {
        "Positive": positive,
        "Neutral": neutral,
        "Negative": negative,
        "Overall": overall
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
