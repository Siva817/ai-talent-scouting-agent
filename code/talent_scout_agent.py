import pandas as pd
import numpy as np
import os
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from sentence_transformers import SentenceTransformer, util


# ================================
# 📁 FILE PATH SETUP (RELATIVE)
# ================================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

RESUME_PATH = os.path.join(BASE_DIR, "data", "resume.csv")
JD_PATH = os.path.join(BASE_DIR, "data", "job_descriptions.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "output", "ranked_candidates.csv")


# ================================
# 📊 LOAD DATA
# ================================
resume_df = pd.read_csv(RESUME_PATH)
jd_df = pd.read_csv(JD_PATH)

print("Resume Columns:", resume_df.columns)
print("JD Columns:", jd_df.columns)


# ================================
# 🧠 TEXT CLEANING
# ================================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

resume_df['clean_resume'] = resume_df['resume'].apply(clean_text)


# ================================
# 🏷️ CATEGORY NORMALIZATION
# ================================
def normalize_category(text):
    return str(text).lower().replace("-", " ").strip()

resume_df['norm_category'] = resume_df['category'].apply(normalize_category)
jd_df['norm_category'] = jd_df['category'].apply(normalize_category)


# ================================
# 🤖 TRAIN CLASSIFIER
# ================================
X = resume_df['clean_resume']
y = resume_df['category']

vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
X_vec = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


# Predict category for all resumes
resume_df['predicted_category'] = model.predict(X_vec)


# ================================
# 🤖 LOAD EMBEDDING MODEL
# ================================
print("\nLoading Sentence Transformer...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

print("Encoding resumes...")
resume_embeddings = embedder.encode(
    resume_df['clean_resume'].tolist(),
    convert_to_tensor=True
)


# ================================
# 🧠 SKILL EXTRACTION (IMPROVED)
# ================================
def extract_skills(text):
    words = text.split()

    stopwords = set([
        "the", "and", "with", "from", "this", "that", "have", "will",
        "your", "about", "their", "been", "were", "into", "using",
        "are", "for", "you", "can", "all", "any", "our", "out",
        "job", "role", "work", "experience"
    ])

    keywords = [w for w in words if len(w) > 4 and w not in stopwords]

    return list(set(keywords))[:20]


# ================================
# 🎯 MATCHING + RANKING
# ================================
results = []

for idx, jd_row in jd_df.iterrows():

    jd_id = jd_row['id']
    jd_text = clean_text(jd_row['jd'])
    jd_category = jd_row['category']
    jd_category_norm = jd_row['norm_category']

    print(f"\nProcessing JD {jd_id} ({jd_category})")

    jd_embedding = embedder.encode(jd_text, convert_to_tensor=True)

    # Filter candidates by category (FIXED)
    candidates = resume_df[
        resume_df['norm_category'] == jd_category_norm
    ]

    if candidates.empty:
        print("⚠️ No candidates found")
        continue

    candidate_indices = candidates.index.tolist()
    candidate_embeddings = resume_embeddings[candidate_indices]

    similarities = util.cos_sim(jd_embedding, candidate_embeddings)[0]

    top_k = min(5, len(candidates))
    top_results = similarities.topk(k=top_k)

    jd_skills = extract_skills(jd_text)

    for rank in range(top_k):
        idx_in_candidates = top_results.indices[rank].item()
        score = top_results.values[rank].item()

        candidate = candidates.iloc[idx_in_candidates]
        candidate_text = candidate['clean_resume']
        candidate_id = candidate['id']

        resume_skills = extract_skills(candidate_text)

        matched = list(set(jd_skills).intersection(resume_skills))

        match_score = min(10, int(score * 10))
        interest_score = min(10, len(matched))

        final_score = round((0.7 * match_score + 0.3 * interest_score), 2)

        result = {
            "jd_id": jd_id,
            "candidate_id": int(candidate_id),
            "predicted_category": candidate['predicted_category'],
            "match_score": match_score,
            "interest_score": interest_score,
            "final_score": final_score,
            "matched_skills": ", ".join(matched),
            "explanation": f"Matched skills: {', '.join(matched)}. Semantic similarity + keyword overlap used."
        }

        print(f"Rank {rank+1}:", result)

        results.append(result)


# ================================
# 💾 SAVE OUTPUT
# ================================
results_df = pd.DataFrame(results)

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
results_df.to_csv(OUTPUT_PATH, index=False)

print(f"\n✅ Results saved at: {OUTPUT_PATH}")
