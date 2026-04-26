# 🤖 AI Talent Scouting & Engagement Agent

## 📌 Overview

This project is an **AI-powered talent scouting system** that matches candidate resumes with job descriptions using:

* Machine Learning (classification)
* NLP (text processing)
* Semantic similarity (embeddings)

It helps recruiters **automatically shortlist and rank candidates** based on relevance and inferred interest.

---

## 🚀 Features

* 📄 Resume classification into job categories
* 🧠 Semantic matching using Sentence Transformers
* 🎯 Candidate ranking based on:

  * Match score (similarity)
  * Interest score (keyword overlap)
* 📊 Final ranked output in CSV format
* 🔍 Explainable results (matched skills + reasoning)

---

## 🧠 How It Works

### 1. Data Processing

* Cleans resume and job description text
* Normalizes categories (e.g., "IT" vs "information-technology")

### 2. Classification Model

* Uses **TF-IDF + Logistic Regression**
* Predicts category of each resume

### 3. Semantic Matching

* Uses **Sentence Transformers (`all-MiniLM-L6-v2`)**
* Converts text into embeddings
* Calculates similarity between resumes and job descriptions

### 4. Scoring Logic

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

### Sample Output Columns:

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
* Model: Logistic Regression (baseline)
* Embeddings: Sentence Transformers

---

## 🎯 Example Use Case

A recruiter uploads:

* Multiple resumes
* A job description

The system:

* Filters relevant candidates
* Ranks top matches
* Explains why they were selected

---

## ⚠️ Limitations

* Basic keyword-based skill extraction
* Limited dataset size
* Some categories may have lower prediction accuracy

---

## 🔮 Future Improvements

* Use advanced models (BERT / LLMs)
* Better skill extraction (NER-based)
* Real-time chatbot integration
* Web UI (Streamlit or Flask)
* Feedback loop for continuous learning

---

## 👨‍💻 Author

Built as a fast prototype for AI-based recruitment automation.

---
