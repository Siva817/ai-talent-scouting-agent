import pandas as pd
import numpy as np
import os
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# PATH SETUP (IMPORTANT)
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

resume_path = os.path.join(ROOT_DIR, "data", "resume.csv")
jd_path = os.path.join(ROOT_DIR, "data", "job_descriptions.csv")
output_path = os.path.join(ROOT_DIR, "output", "ranked_candidates.csv")

# -------------------------------
# LOAD DATA
# -------------------------------
resume_df = pd.read_csv(resume_path)
jd_df = pd.read_csv(jd_path)

print("Columns:", resume_df.columns)
print("\nCategory distribution:\n", resume_df['category'].value_counts())
print("\nMissing values:\n", resume_df.isnull().sum())

# -------------------------------
# TRAIN CLASSIFICATION MODEL
# -------------------------------
X = resume_df['resume']
y = resume_df['category']

tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
X_vec = tfidf.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Predict categories for all resumes
resume_df['predicted_category'] = model.predict(X_vec)

# -------------------------------
# LOAD EMBEDDING MODEL
# -------------------------------
print("\nLoading Sentence Transformer...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

print("Encoding resumes...")
resume_embeddings = embedder.encode(
    resume_df['resume'].tolist(), show_progress_bar=True
)

# -------------------------------
# HELPER FUNCTIONS
# -------------------------------

def extract_keywords(text):
    words = re.findall(r'\b\w+\b', text.lower())
    return list(set(words))


def compute_match_score(similarity):
    return int(np.clip(round(similarity * 10), 1, 10))


def compute_interest_score(resume_text, jd_text):
    resume_words = set(extract_keywords(resume_text))
    jd_words = set(extract_keywords(jd_text))

    overlap = len(resume_words & jd_words)
    score = min(10, max(1, overlap // 5 + 5))  # simple heuristic
    return score


def get_matched_skills(resume_text, jd_text):
    resume_words = set(extract_keywords(resume_text))
    jd_words = set(extract_keywords(jd_text))

    matched = list(resume_words & jd_words)
    return ", ".join(matched[:5]) if matched else "limited overlap"


def generate_explanation(skills):
    return f"Matched skills: {skills}. Semantic similarity contributed to match score. Interest score derived from resume signals."

# -------------------------------
# PROCESS EACH JD
# -------------------------------
all_results = []

print("\nLoaded JDs:", jd_df.shape)

for idx, jd_row in jd_df.iterrows():
    jd_id = jd_row['id']
    jd_text = jd_row['job_description']
    jd_category = jd_row['category']

    print("\n==============================")
    print(f"Processing JD {jd_id} ({jd_category})")

    # Filter candidates by predicted category
    filtered = resume_df[
        resume_df['predicted_category'] == jd_category
    ].copy()

    if filtered.empty:
        print("⚠️ No candidates found")
        continue

    # Encode JD
    jd_embedding = embedder.encode([jd_text])

    # Compute similarity
    sims = cosine_similarity(jd_embedding, resume_embeddings)[0]
    filtered['similarity'] = sims[filtered.index]

    # Scores
    filtered['match_score'] = filtered['similarity'].apply(compute_match_score)
    filtered['interest_score'] = filtered['resume'].apply(
        lambda x: compute_interest_score(x, jd_text)
    )

    filtered['final_score'] = (
        0.7 * filtered['match_score'] +
        0.3 * filtered['interest_score']
    ).round().astype(int)

    # Explainability
    filtered['matched_skills'] = filtered['resume'].apply(
        lambda x: get_matched_skills(x, jd_text)
    )

    filtered['explanation'] = filtered['matched_skills'].apply(
        generate_explanation
    )

    # Sort
    top_candidates = filtered.sort_values(
        by='final_score', ascending=False
    ).head(5)

    print("\nTop 5 Candidates:")

    for rank, (_, row) in enumerate(top_candidates.iterrows(), 1):
        result = {
            "jd_id": jd_id,
            "candidate_id": int(row['id']),
            "predicted_category": row['predicted_category'],
            "match_score": int(row['match_score']),
            "interest_score": int(row['interest_score']),
            "final_score": int(row['final_score']),
            "matched_skills": row['matched_skills'],
            "explanation": row['explanation']
        }

        print(f"Rank {rank}:", result)
        all_results.append(result)

# -------------------------------
# SAVE RESULTS
# -------------------------------
results_df = pd.DataFrame(all_results)
results_df.to_csv(output_path, index=False)

print(f"\n✅ Results saved at: {output_path}")
