from flask import Flask, render_template, request
import pandas as pd
import ollama
import os
from collections import Counter
from tqdm import tqdm

app = Flask(__name__)

def analyze_feedback(text):
    """Analyze individual feedback using Mistral model and return structured data."""
    prompt = f"""
    Analyze the following feedback:

    "{text}"

    Return output in EXACTLY this structured format:

    Overall Sentiment: (Positive/Neutral/Negative)

    Overall Insight: 
    (One short paragraph summarizing key strengths and weaknesses)

    Key Factors:
    
    Negative:
    - (Short bullet points listing negative aspects)
    
    Positive:
    - (Short bullet points listing positive aspects)
    
    Neutral:
    - (Short bullet points listing neutral aspects, or 'N/A' if none)
    """

    response = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}])
    return response['message']['content']

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == "":
            return "No file uploaded", 400
        
        file_path = "feedbacks.csv"
        file.save(file_path)
        return analyze_feedbacks(file_path)

    return render_template("index.html")

def analyze_feedbacks(file_path):
    df = pd.read_csv(file_path)
    if "FEEDBACK" not in df.columns:
        return "CSV must contain a 'FEEDBACK' column", 400

    sentiment_counts = Counter()
    insights = []
    all_positive, all_negative, all_neutral = set(), set(), set()

    for feedback in tqdm(df["FEEDBACK"].dropna(), desc="Processing Feedbacks", unit="feedback"):
        analysis = analyze_feedback(feedback)

        try:
            sentiment = analysis.split("Overall Sentiment:")[1].split("\n")[0].strip()
            insight = analysis.split("Overall Insight:")[1].split("\nKey Factors:")[0].strip()
            insights.append(insight)

            negative = analysis.split("Negative:")[1].split("Positive:")[0].strip().split("\n- ")
            positive = analysis.split("Positive:")[1].split("Neutral:")[0].strip().split("\n- ")
            neutral = analysis.split("Neutral:")[1].strip().split("\n- ")

            sentiment_counts[sentiment] += 1

            all_negative.update([n.strip() for n in negative if n.strip() and "N/A" not in n])
            all_positive.update([p.strip() for p in positive if p.strip() and "N/A" not in p])
            all_neutral.update([neu.strip() for neu in neutral if neu.strip() and "N/A" not in neu])

        except IndexError:
            continue

    overall_sentiment = sentiment_counts.most_common(1)[0][0]
    summary_insight = " ".join(insights[:3])  # Shorten to first 3 insights

    return render_template("result.html", sentiment=overall_sentiment, 
                           insight=summary_insight[:250],  # Limit insight to ~3-4 lines
                           positive=sorted(all_positive), 
                           negative=sorted(all_negative), 
                           neutral=sorted(all_neutral))

def extract_value(text, start_marker, end_marker=None):
    """Extracts a single-line value from the structured response."""
    try:
        if end_marker:
            return text.split(start_marker)[1].split(end_marker)[0].strip()
        return text.split(start_marker)[1].strip()
    except IndexError:
        return "N/A"

def extract_list(text, start_marker, end_marker=None):
    """Extracts a list of items from the structured response."""
    try:
        section = extract_value(text, start_marker, end_marker)
        return {item.strip() for item in section.split("\n- ") if item.strip() and "N/A" not in item}
    except IndexError:
        return set()


if __name__ == "__main__":
    app.run(debug=True)
