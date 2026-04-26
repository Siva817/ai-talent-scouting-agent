# 🤖 AI Talent Scouting & Engagement Agent

## 📌 Overview

This project is an AI-powered recruitment system that automatically:

* Classifies resumes into job categories
* Matches resumes with job descriptions
* Ranks candidates based on relevance and inferred interest

It combines **Machine Learning + NLP + Semantic Search** to automate early-stage hiring.

---

## 🚀 Features

* 📄 Resume classification using TF-IDF + Logistic Regression
* 🧠 Semantic matching using Sentence Transformers
* 🎯 Intelligent candidate ranking system
* 🔍 Explainable results (matched skills + reasoning)
* 📊 CSV-based output for easy integration

---

## 🏗️ Architecture

The system follows a hybrid NLP pipeline:

```id="a8p3qv"
Resumes CSV
   ↓
Text Cleaning & Preprocessing
   ↓
TF-IDF Vectorization
   ↓
Logistic Regression Classifier
   ↓
Predicted Resume Category

Job Descriptions CSV
   ↓
Text Cleaning
   ↓
Sentence Transformer Embeddings

Resumes → Sentence Transformer Embeddings
   ↓
Cosine Similarity Matching with Job Descriptions
   ↓
Filter by Predicted Category
   ↓
Scoring Engine
   ↓
Ranked Candidates Output (CSV)
```

---

## 🧠 How It Works

### 1. Data Preprocessing

* Removes special characters and noise
* Converts text to lowercase
* Standardizes resume and job description format

---

### 2. Resume Classification

* Uses **TF-IDF vectorization**
* Logistic Regression model
* Predicts job category of each resume

---

### 3. Semantic Matching

* Uses Sentence Transformer (`all-MiniLM-L6-v2`)
* Converts resumes and job descriptions into embeddings
* Uses cosine similarity for semantic matching

---

## 🎯 Scoring Logic

Each candidate is ranked using a weighted scoring system:

### 1. Match Score (70%)

* Based on cosine similarity between resume and job description embeddings
* Captures semantic relevance and contextual similarity

### 2. Interest Score (30%)

* Based on keyword overlap between resume and job description
* Captures domain-specific skills and intent

---

### 📌 Final Score Formula:

```id="sc3r1a"
Final Score = 0.7 × Match Score + 0.3 × Interest Score
```

---

## 📂 Project Structure

```id="d8v1qp"
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

```bash id="k2n9hf"
pip install pandas numpy scikit-learn sentence-transformers
```

### 2. Run the project

```bash id="v7c1mq"
cd code
python talent_scout_agent.py
```

---

## 📥 Input Format

### resume.csv

```id="r2x9ka"
id, resume, category
```

### job_descriptions.csv

```id="p4m8zd"
id, category, jd
```

---

## 📤 Output

Generated file:

```id="o9q2ls"
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

1. Recruiter uploads resumes and job descriptions
2. System classifies resumes into domains
3. Semantic matching finds relevant candidates
4. Candidates are ranked and explained

---

## ⚠️ Limitations

* Basic keyword-based skill extraction
* Dataset imbalance affects performance
* Some categories have lower prediction accuracy

---

## 🔮 Future Improvements

* Upgrade to transformer-based classification (BERT/LLM)
* Advanced skill extraction using NER models
* Web UI using Streamlit or Flask
* Real-time recruitment dashboard
* Feedback loop for continuous learning

---

## 👨‍💻 Author

Built as an AI recruitment automation prototype using NLP and Machine Learning.

---
