from flask import Flask, render_template, request, jsonify
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
    (One very short paragraph summarizing key strengths and weaknesses)

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

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "").strip()

    if not user_message:
        return jsonify({"reply": "Please enter a valid message."})

    # Define chatbot prompt for structured and bold-formatted replies
    prompt = f"""
    You are Stack AI, a business analyst chatbot.
    Format the response so that:
    - Headings are in bold (<b>Title<b>).
    - Subheadings are also bold but inside <b> (- <b>Subheading<b>).
    - Each point appears on a new line.
    - Content is short and structured.
    - Avoid long paragraphs.

    User: {user_message}
    Chatbot:
    """

    # Call Ollama AI model (Mistral) for response
    response = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}])

    chatbot_reply = response.get("message", {}).get("content", "I'm not sure how to answer that.")

    # Ensure proper formatting with bold headings and subheadings
    formatted_reply = chatbot_reply.replace("\n", "\n\n")  # Improve spacing

    return jsonify({"reply": formatted_reply})

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        feedback_text = request.form.get("feedback_text")
        file = request.files.get("file")

        if file and file.filename != "":
            file_path = "feedbacks.csv"
            file.save(file_path)
            return analyze_feedbacks(file_path)

        elif feedback_text.strip():
            analysis = analyze_feedback(feedback_text)

            # Extract sentiment as a single word
            sentiment = extract_value(analysis, "Overall Sentiment:").split()[0]  

            insight = extract_value(analysis, "Overall Insight:", "Key Factors:")
            positive = extract_list(analysis, "Positive:", "Negative:")
            negative = extract_list(analysis, "Negative:", "Neutral:")
            neutral = extract_list(analysis, "Neutral:")

            return render_template("result.html", 
                                   sentiment=sentiment, 
                                   insight=insight, 
                                   positive=positive, 
                                   negative=negative, 
                                   neutral=neutral)

        else:
            return "Please enter feedback text or upload a CSV file", 400

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
            sentiment = extract_value(analysis, "Overall Sentiment:")
            insight = extract_value(analysis, "Overall Insight:", "Key Factors:")
            insights.append(insight)

            negative = extract_list(analysis, "Negative:", "Positive:")
            positive = extract_list(analysis, "Positive:", "Neutral:")
            neutral = extract_list(analysis, "Neutral:")

            sentiment_counts[sentiment] += 1

            all_negative.update(negative)
            all_positive.update(positive)
            all_neutral.update(neutral)

        except IndexError:
            continue

    overall_sentiment = sentiment_counts.most_common(1)[0][0]
    summary_insight = " ".join(insights[:3])  # Shorten to first 3 insights

    return render_template("result.html", sentiment=overall_sentiment, 
                           insight=summary_insight[:250], 
                           positive=sorted(all_positive), 
                           negative=sorted(all_negative), 
                           neutral=sorted(all_neutral))

def extract_value(text, start_marker, end_marker=None):
    """Extracts a single-line value from the structured response."""
    try:
        if end_marker:
            return text.split(start_marker)[1].split(end_marker)[0].strip()
        return text.split(start_marker)[1].split("\n")[0].strip()  # Ensure single-word sentiment
    except IndexError:
        return "N/A"

def extract_list(text, start_marker, end_marker=None):
    """Extracts a list of items from the structured response."""
    try:
        section = extract_value(text, start_marker, end_marker)
        return [item.strip() for item in section.split("\n") if item.strip().startswith("- ")]
    except IndexError:
        return []

if __name__ == "__main__":
    app.run(debug=True)