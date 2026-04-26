# 🤖 AI Talent Scouting & Engagement Agent

## 📌 Overview

This project is an AI-powered recruitment system that automatically:

* Classifies resumes into job categories
* Matches resumes with job descriptions
* Ranks candidates based on relevance and inferred interest

It combines **Machine Learning + NLP + Semantic Search** to reduce manual hiring effort.

---

## 🚀 Features

* 📄 Resume classification using TF-IDF + Logistic Regression
* 🧠 Semantic similarity using Sentence Transformers
* 🎯 Candidate ranking based on:

  * Match Score (semantic similarity)
  * Interest Score (keyword overlap)
* 📊 Final ranked output in CSV format
* 🔍 Explainable results (matched skills + reasoning)

---

## 🏗️ Architecture

The system follows a hybrid NLP pipeline:

```
Resumes CSV
   ↓
Text Cleaning & Preprocessing
   ↓
TF-IDF Vectorization
   ↓
Logistic Regression Classifier
   ↓
Predicted Job Category

Job Descriptions CSV
   ↓
Text Cleaning
   ↓
Sentence Transformer Embeddings

Resumes → Sentence Transformer Embeddings
   ↓
Cosine Similarity Matching with Job Descriptions
   ↓
Filtering by Predicted Category
   ↓
Scoring System:
   - Match Score (semantic similarity)
   - Interest Score (keyword overlap)
   ↓
Final Ranking Engine
   ↓
Output: ranked_candidates.csv
```

---

## 🧠 How It Works

### 1. Data Preprocessing

* Cleans resume and job description text
* Removes special characters and noise

### 2. Resume Classification

* Uses TF-IDF vectorization
* Logistic Regression predicts job category

### 3. Semantic Matching

* Sentence Transformer (`all-MiniLM-L6-v2`)
* Converts resumes and JDs into embeddings
* Uses cosine similarity for matching

### 4. Scoring System

Final Score =
`0.7 × Match Score (semantic similarity)` +
`0.3 × Interest Score (keyword overlap)`

---

## 📂 Project Structure

```
ai-talent-scouting-agent/
│
├── code/
│   └── talent_scout_agent.py
│
├── data/
│   ├── resume.csv
│   └── job_descriptions.csv
│
├── output/
│   └── ranked_candidates.csv
│
└── README.md
```

---

## ▶️ How to Run

### 1. Install dependencies

```bash
pip install pandas numpy scikit-learn sentence-transformers
```

### 2. Run the project

```bash
cd code
python talent_scout_agent.py
```

---

## 📥 Input Format

### resume.csv

```
id, resume, category
```

### job_descriptions.csv

```
id, category, jd
```

---

## 📤 Output

The system generates:

```
output/ranked_candidates.csv
```

### Output Columns:

* jd_id
* candidate_id
* predicted_category
* match_score
* interest_score
* final_score
* matched_skills
* explanation

---

## 📊 Model Performance

* Accuracy: ~68–70%
* Model: Logistic Regression
* Embeddings: Sentence Transformers (MiniLM)

---

## 🎯 Example Use Case

1. Recruiter uploads resumes + job descriptions
2. System classifies resumes into domains
3. Semantic matching finds relevant candidates
4. Candidates are ranked and explained

---

## ⚠️ Limitations

* Basic keyword-based skill extraction
* Dataset size affects accuracy
* Some categories may be underrepresented

---

## 🔮 Future Improvements

* Upgrade to transformer-based classifier (BERT/LLM)
* Better skill extraction using NER models
* Web UI (Streamlit/Flask)
* Real-time recruitment dashboard
* Feedback loop for continuous learning

---

## 👨‍💻 Author

Built as a prototype AI recruitment assistant using NLP and Machine Learning.

---
