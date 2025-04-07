import os
import json
import google.generativeai as genai
import numpy as np
import spacy
import requests
import streamlit as st
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import secrets

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Generative AI
genai.configure(api_key=GEMINI_API_KEY)

# Flask App
app = Flask(__name__)

# Download & Extract Model if Not Exists
MODEL_DIR = "all-MiniLM-L6-v2"
GDRIVE_FILE_ID = "1yQTnfOqAzynnAE7JNt0YJOxPrai37VaE"
MODEL_ZIP = "model.zip"

if not os.path.exists(MODEL_DIR):
    print("Downloading model from Google Drive...")
    os.system(f"wget --no-check-certificate 'https://drive.google.com/uc?export=download&id={GDRIVE_FILE_ID}' -O {MODEL_ZIP}")
    os.system(f"unzip {MODEL_ZIP} && rm {MODEL_ZIP}")

# Load SentenceTransformer Model
bert_model = SentenceTransformer(MODEL_DIR)

# Load NLP Model
nlp = spacy.load("en_core_web_sm")

# SHL Assessments Data
shl_assessments = [
    {
        "id": "SHL001",
        "name": "Verify Cognitive Ability",
        "description": "Measures verbal, numerical, and logical reasoning abilities",
        "type": "Cognitive Ability",
        "tags": ["reasoning", "aptitude", "critical thinking"],
        "target_roles": ["Entry-level", "Analyst", "Engineer"],
        "target_industries": ["Technology", "Finance", "Consulting"],
        "difficulty_level": "Intermediate",
        "duration_minutes": 45,
        "languages": ["English", "Spanish", "French"],
        "url": "https://www.shl.com/products/verify-cognitive/",
        "remote_testing": True,
        "adaptive_support": True
    }
]

# Encode SHL Assessments
assessment_texts = [a["name"] + " " + a["description"] for a in shl_assessments]
assessment_vectors = bert_model.encode(assessment_texts)

# Extract Keywords from Job Description
def extract_keywords(text):
    doc = nlp(text)
    return [token.lemma_.lower() for token in doc if token.is_alpha and not token.is_stop]

# Analyze Job Description with GPT
def analyze_with_gpt(job_description):
    try:
        model = genai.GenerativeModel("gemini-1.5-pro-latest")
        prompt = f"Analyze this job description and suggest required skills and best SHL assessments. Job description: {job_description}"
        response = model.generate_content(prompt)
        return response.text if response else "No response from AI"
    except Exception as e:
        return f"AI Analysis Failed: {str(e)}"

# Flask API for Recommendations
@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    job_description = data.get("job_description", "")
    keywords = extract_keywords(job_description)
    ai_analysis = analyze_with_gpt(job_description)
    job_vector = bert_model.encode([job_description])[0]
    similarities = cosine_similarity([job_vector], assessment_vectors)[0]
    ranked_indices = np.argsort(similarities)[::-1][:5]
    recommendations = [{"name": shl_assessments[i]["name"], 
                        "score": float(similarities[i]), 
                        "url": shl_assessments[i]["url"],
                        "duration": shl_assessments[i].get("duration_minutes", "Unknown"),
                        "test_type": shl_assessments[i].get("type", "N/A"),
                        "remote": shl_assessments[i].get("remote_testing", "N/A"),
                        "adaptive": shl_assessments[i].get("adaptive_support", "N/A")
                       } for i in ranked_indices]
    return jsonify({"recommendations": recommendations, "ai_analysis": ai_analysis, "keywords": keywords})

# Streamlit Frontend
def frontend():
    st.set_page_config(page_title="SHL Assessment Recommender", page_icon="🔍", layout="wide")
    st.title("🔍 SHL Assessment Recommendation System")
    st.write("AI-powered tool to recommend SHL assessments based on job descriptions.")
    job_description = st.text_area("📌 Paste Job Description Below:", height=150)
    if st.button("🚀 Get Recommendations"):
        with st.spinner("Processing... Please wait ⏳"):
            response = requests.post("http://127.0.0.1:5000/recommend", json={"job_description": job_description})
        if response.status_code == 200:
            data = response.json()
            st.subheader("🔬 AI Analysis")
            st.markdown(f"**{data['ai_analysis']}**")
            st.subheader("🔑 Extracted Keywords")
            st.write(", ".join(data["keywords"]))
            st.subheader("📋 Recommended Assessments")
            for i, rec in enumerate(data["recommendations"]):
                st.write(f"**{i+1}. [{rec['name']}]({rec['url']})**")
                st.write(f"⏳ Duration: {rec['duration']} min | 🧠 Type: {rec['test_type']}")
                st.write(f"✅ Remote Testing: {rec['remote']} | 🔄 Adaptive Support: {rec['adaptive']}")
                st.write(f"⭐ Match Score: {rec['score']:.2f}")
                st.write("---")
        else:
            st.error("❌ Error fetching recommendations. Please try again.")

# Run Flask and Streamlit in Parallel
if __name__ == '__main__':
    import threading
    from werkzeug.serving import run_simple
    flask_thread = threading.Thread(target=lambda: run_simple('127.0.0.1', 5000, app))
    flask_thread.start()
    frontend()

