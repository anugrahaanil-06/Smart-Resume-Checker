import streamlit as st
import PyPDF2
import docx
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

model = SentenceTransformer("all-MiniLM-L6-v2")


def extract_text(file):
    if file.name.endswith(".pdf"):
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text

    elif file.name.endswith(".docx"):
        doc = docx.Document(file)
        return " ".join([p.text for p in doc.paragraphs])

    elif file.name.endswith(".txt"):
        return file.read().decode("utf-8")


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z ]", "", text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)


def cosine_similarity_score(resume, jd):
    tfidf = TfidfVectorizer()
    vectors = tfidf.fit_transform([resume, jd])
    score = cosine_similarity(vectors[0:1], vectors[1:2])
    return round(score[0][0] * 100, 2)


def semantic_similarity_score(resume, jd):
    emb1 = model.encode(resume, convert_to_tensor=True)
    emb2 = model.encode(jd, convert_to_tensor=True)
    score = util.cos_sim(emb1, emb2)
    return round(float(score[0][0]) * 100, 2)


def keyword_match_score(resume, jd):
    resume_words = set(resume.split())
    jd_words = set(jd.split())
    matched = resume_words.intersection(jd_words)
    return round((len(matched) / len(jd_words)) * 100, 2)


st.set_page_config(page_title="Smart Resume Checker", layout="centered")

st.title("Smart Resume Checker")
st.subheader("ATS-based Resume Matching using NLP")

st.write("Upload a resume and paste the job description to check compatibility.")

resume_file = st.file_uploader("Upload Resume (PDF / DOCX / TXT)", type=["pdf", "docx", "txt"])
job_description = st.text_area("Paste Job Description Here")

if st.button("Check Resume Match"):
    if resume_file and job_description.strip() != "":
        resume_text = extract_text(resume_file)
        resume_clean = clean_text(resume_text)
        jd_clean = clean_text(job_description)

        cosine_score = cosine_similarity_score(resume_clean, jd_clean)
        semantic_score = semantic_similarity_score(resume_clean, jd_clean)
        keyword_score = keyword_match_score(resume_clean, jd_clean)

        final_score = round((cosine_score + semantic_score + keyword_score) / 3, 2)

        st.success("Resume Analysis Completed")

        st.metric("Cosine Similarity (%)", cosine_score)
        st.metric("Semantic Similarity (%)", semantic_score)
        st.metric("Keyword Match (%)", keyword_score)

        st.subheader(f"Overall Resume Match Score: {final_score}%")

        if final_score >= 75:
            st.success("Strong match! Resume fits the job description well.")
        elif final_score >= 50:
            st.warning("Moderate match. Resume needs improvement.")
        else:
            st.error("Low match. Resume does not align with the job description.")

    else:
        st.error("Please upload a resume and enter job description.")
