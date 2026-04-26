import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ==============================
# 1. LOAD DATA
# ==============================

resume_df = pd.read_csv(
    r"C:\Users\prasa\OneDrive\Desktop\talent_scouting_and_engagement_agent\resume.csv"
)
jd_df = pd.read_csv(
    r"C:\Users\prasa\OneDrive\Desktop\talent_scouting_and_engagement_agent\job_descriptions.csv"
)

# Normalize category names
def clean_category(text):
    return text.lower().replace("-", " ").strip()

resume_df['category'] = resume_df['category'].apply(clean_category)
jd_df['category'] = jd_df['category'].apply(clean_category)

print("Columns:", resume_df.columns)

# ==============================
# 2. TRAIN CLASSIFICATION MODEL
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    resume_df['resume'], resume_df['category'], test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

clf = LogisticRegression(max_iter=200)
clf.fit(X_train_vec, y_train)

y_pred = clf.predict(X_test_vec)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ==============================
# 3. LOAD EMBEDDING MODEL
# ==============================

print("\nLoading Sentence Transformer...")
model = SentenceTransformer('all-MiniLM-L6-v2')

print("Encoding resumes...")
resume_embeddings = model.encode(
    resume_df['resume'].tolist(), batch_size=32, show_progress_bar=True
)

# ==============================
# 4. SKILL EXTRACTION
# ==============================

def extract_keywords(text):
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    return list(set(words))

# ==============================
# 5. SCORING FUNCTIONS
# ==============================

def scale_score(value):
    """Convert similarity (0–1) to score (1–10)"""
    return int(np.clip(round(value * 10), 1, 10))


def compute_interest(resume_text):
    """Simple heuristic for interest"""
    resume_text = resume_text.lower()

    if "looking" in resume_text or "seeking" in resume_text:
        return 8
    elif "experienced" in resume_text:
        return 7
    else:
        return 6


def explain_match(jd_skills, resume_text):
    resume_words = set(resume_text.lower().split())
    matched = list(set(jd_skills) & resume_words)

    if matched:
        return ", ".join(matched[:5])
    else:
        return "general relevance"

# ==============================
# 6. MATCHING PIPELINE
# ==============================

results = []

print("\nLoaded JDs:", jd_df.shape)

for _, jd_row in jd_df.iterrows():

    jd_id = int(jd_row['id'])
    jd_text = jd_row['jd']
    jd_category = jd_row['category']

    print("\n==============================")
    print(f"Processing JD {jd_id} ({jd_category})")

    jd_vec = vectorizer.transform([jd_text])
    predicted_category = clean_category(clf.predict(jd_vec)[0])

    jd_embedding = model.encode([jd_text])[0]

    jd_skills = extract_keywords(jd_text)

    # Filter candidates
    candidates = resume_df[
        (resume_df['category'] == jd_category) |
        (resume_df['category'] == predicted_category)
    ]

    if len(candidates) == 0:
        print("⚠️ No exact category match → using all resumes")
        candidates = resume_df

    candidate_indices = candidates.index.tolist()
    candidate_embeddings = resume_embeddings[candidate_indices]

    similarities = cosine_similarity([jd_embedding], candidate_embeddings)[0]

    scored_candidates = []

    for idx, sim in zip(candidate_indices, similarities):

        resume_text = resume_df.loc[idx, 'resume']
        candidate_id = int(resume_df.loc[idx, 'id'])
        category = resume_df.loc[idx, 'category']

        match_score = scale_score(sim)
        interest_score = compute_interest(resume_text)

        final_score = int(round((match_score + interest_score) / 2))

        matched_skills = explain_match(jd_skills, resume_text)

        explanation = (
            f"Skills match: {matched_skills}. "
            f"Semantic similarity used for matching. "
            f"Interest inferred from resume tone."
        )

        scored_candidates.append({
            "jd_id": jd_id,
            "candidate_id": candidate_id,
            "category": category,
            "match_score": match_score,
            "interest_score": interest_score,
            "final_score": final_score,
            "matched_skills": matched_skills,
            "explanation": explanation
        })

    # Sort top 5
    top_candidates = sorted(
        scored_candidates,
        key=lambda x: x['final_score'],
        reverse=True
    )[:5]

    print("\nTop 5 Candidates:")

    for i, cand in enumerate(top_candidates, 1):
        print(f"Rank {i}: {cand}")
        results.append(cand)

# ==============================
# 7. SAVE RESULTS
# ==============================

output_path = r"C:/Users/prasa/OneDrive/Desktop/talent_scouting_and_engagement_agent/ranked_candidates.csv"

pd.DataFrame(results).to_csv(output_path, index=False)

print(f"\n✅ Results saved at: {output_path}")
