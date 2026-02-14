# Smart-Resume-Checker
# AI-Powered ATS Resume Matching System using NLP

Smart Resume Checker is a Natural Language Processing (NLP) based web application that evaluates the compatibility between a resume and a job description. The system leverages multiple similarity techniques to generate an overall match score, simulating how modern Applicant Tracking Systems (ATS) analyze resumes.

This project demonstrates practical implementation of text preprocessing, feature extraction, and semantic similarity using transformer-based models.

# Key Features

* Resume upload support (PDF, DOCX, TXT)

* Automated text extraction and preprocessing

* Multi-level similarity analysis:

* TF-IDF Cosine Similarity

* Transformer-based Semantic Similarity

* Keyword Matching Score

* Consolidated Resume Match Percentage

* Intelligent feedback based on match strength

* Clean and interactive Streamlit UI

# Technical Implementation

The system follows a structured NLP pipeline:

1. Text Extraction

* PyPDF2 for PDF parsing

* python-docx for DOCX processing

2. Text Preprocessing

* Lowercasing

* Special character removal

* Stopword removal using NLTK

3. Similarity Computation

* TF-IDF Vectorization (Scikit-learn)

* Cosine Similarity

* Sentence Transformers (all-MiniLM-L6-v2) for semantic embeddings

* Keyword overlap analysis

4. Final Score Calculation

* Weighted average of all similarity metrics

* Score interpretation (Strong / Moderate / Low Match)

# Tech Stack

* Python

* Streamlit

* Scikit-learn

* NLTK

* Sentence-Transformers

* PyPDF2

* python-docx

# Project Structure
smart-resume-checker/


├── app.py              # Main Streamlit application

├── requirements.txt    # Project dependencies

└── README.md           # Documentation

# Installation & Setup
# Clone the Repository
git clone https://github.com/anugrahaanil-06/smart-resume-checker.git
cd smart-resume-checker

# Install Dependencies
pip install -r requirements.txt

# Run the Application
streamlit run app.py


The application will launch at:
http://localhost:8501

# Practical Applications

* Resume optimization for job seekers

* ATS compatibility testing

* NLP portfolio demonstration

* Learning project for machine learning and text similarity

# Future Enhancements

* Skill gap detection and recommendation system

* Section-wise resume evaluation

* Downloadable PDF analysis report

* Deployment on cloud platforms

* Model fine-tuning for domain-specific roles

# Author

Anugraha AL

Entry-Level AI/ML Engineer | Data Analyst | Python Developer  
B.Sc. Physics Graduate focused on NLP, Machine Learning, and Intelligent Systems.
