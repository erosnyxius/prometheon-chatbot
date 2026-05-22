from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json

app = Flask(__name__)

# Load AI model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load FAQ data
with open("faq.json", "r", encoding="utf-8") as f:
    faq_data = json.load(f)

questions = [item["question"] for item in faq_data]
answers = [item["answer"] for item in faq_data]

# Convert FAQ questions into vectors
question_vectors = model.encode(questions)

@app.route("/")
def home():
    return "MyselfBOT is running!"

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("message", "")

    # Convert user message to vector
    user_vector = model.encode([user_input])

    # Compare similarities
    similarities = cosine_similarity(user_vector, question_vectors)

    best_match_index = similarities.argmax()
    best_score = similarities[0][best_match_index]

    # Confidence threshold
    if best_score > 0.5:
        response = answers[best_match_index]
    else:
        response = "Sorry, I don't understand that yet."

    return jsonify({
        "reply": response,
        "score": float(best_score)
    })

if __name__ == "__main__":
    app.run(debug=True)