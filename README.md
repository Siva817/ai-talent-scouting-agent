# AI-Powered Talent Scouting & Engagement Agent

## Overview
This project helps recruiters automatically find and rank candidates based on job descriptions.

## Features
- Resume classification using ML
- Semantic matching using Sentence Transformers
- Candidate ranking based on:
  - Match Score (1–10)
  - Interest Score (1–10)
- Explainability:
  - Matched skills
  - Reasoning for each candidate

## How to Run

1. Install dependencies:
```
pip install pandas scikit-learn sentence-transformers
```

2. Place dataset:
- resume.csv inside project folder

3. Run:
```
python checkcheckcheck.py
```

4. Output:
- ranked_candidates.csv

## Architecture

JD Input → Parsing → ML Classification → Semantic Matching → Scoring → Ranking → Output

## Scoring Logic

Final Score = (0.7 × Match Score) + (0.3 × Interest Score)

## Sample Output

- Candidate ID: 20674668  
- Match Score: 5  
- Interest Score: 5  
- Final Score: 6  
- Explanation: Strong semantic similarity and relevant skills
