# app.py
import streamlit as st
import pandas as pd
import numpy as np
import io
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Try to import pdfplumber for PDF text extraction
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except Exception:
    PDFPLUMBER_AVAILABLE = False

# ---------------------------
# Helper functions
# ---------------------------

def extract_text_from_pdf(file_bytes):
    if not PDFPLUMBER_AVAILABLE:
        return ""
    text = []
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)
    except Exception:
        return ""
    return "\n".join(text)

def extract_text_from_txt(file_bytes):
    try:
        return file_bytes.decode('utf-8', errors='ignore')
    except Exception:
        return ""

def normalize_text(text):
    # basic normalization: lowercase, remove punctuation/digits, trim whitespace
    if not text:
        return ""
    text = text.lower()
    # keep alphabets and spaces
    text = re.sub(r'[^a-z\s]', ' ', text)
    # collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_default_job_roles():
    # A sample job-role dataset. Replace with CSV or DB in production.
    data = [
        {"role": "Data Analyst", "description": "SQL, Excel, Python, pandas, data visualization, Tableau, reporting"},
        {"role": "Machine Learning Engineer", "description": "Python, scikit-learn, TensorFlow, PyTorch, feature engineering, model deployment"},
        {"role": "Backend Developer (Python)", "description": "Python, Django, Flask, REST APIs, SQL, PostgreSQL, unit testing"},
        {"role": "DevOps Engineer", "description": "CI/CD, Docker, Kubernetes, AWS, monitoring, Terraform, scripting"},
        {"role": "QA Engineer / Test Automation", "description": "Selenium, Python, pytest, test automation frameworks, CI"},
        {"role": "Full Stack Developer", "description": "React, JavaScript, Python, Django/Flask, REST, database design"},
        {"role": "Business Analyst", "description": "requirements gathering, stakeholder communication, Excel, PowerPoint, SQL basics"},
        {"role": "NLP Engineer", "description": "Python, spaCy, transformers, text preprocessing, language models, tokenization"},
    ]
    return pd.DataFrame(data)

def recommend_roles(resume_text, jobs_df, top_n=5):
    # build corpus: resume + job descriptions
    docs = [normalize_text(resume_text)] + [normalize_text(d) for d in jobs_df['description'].tolist()]

    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X = vectorizer.fit_transform(docs)

    resume_vec = X[0:1]
    job_vecs = X[1:]
    sims = cosine_similarity(resume_vec, job_vecs).flatten()
    jobs_df = jobs_df.copy()
    jobs_df['score'] = sims
    jobs_df = jobs_df.sort_values('score', ascending=False).reset_index(drop=True)
    return jobs_df.head(top_n), vectorizer, X

def highlight_keywords(resume_text, job_desc, top_k=8):
    """
    Return top overlapping words between resume_text and job_desc based on simple set intersection.
    More advanced: compute top TF-IDF features overlap.
    """
    r_tokens = set(normalize_text(resume_text).split())
    j_tokens = set(normalize_text(job_desc).split())
    common = list(r_tokens & j_tokens)
    common = sorted(common, key=lambda s: len(s), reverse=True)  # longer tokens first
    return common[:top_k]

# ---------------------------
# Streamlit UI
# ---------------------------

st.set_page_config(page_title="AI Resume Recommender", layout="wide",
                   initial_sidebar_state="expanded")

# Header and styling
st.markdown("""
    <style>
    .stApp { background: linear-gradient(180deg, #f8fafc 0%, #ffffff 100%); }
    .title { font-size: 30px; font-weight:700; color:#0f172a; }
    .subtitle { color:#334155; font-size:14px; margin-bottom: 8px; }
    .card { background: #ffffff; border-radius: 10px; padding: 18px; box-shadow: 0 4px 14px rgba(20, 20, 20, 0.04); }
    .role { font-weight:700; font-size:16px }
    .score { color: #065f46; font-weight:700; }
    .small { color:#475569; font-size:13px; }
    </style>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1.2], gap="small")

with col1:
    st.markdown("<div class='title'>AI-Powered Resume Recommender</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Upload your resume to get the top job roles that match your skills.</div>", unsafe_allow_html=True)

    st.markdown("### 1) Upload Resume (PDF or TXT)")
    uploaded_file = st.file_uploader("", type=["pdf", "txt"])

    st.markdown("### 2) Job roles dataset")
    st.markdown("You can use the built-in sample roles or upload a CSV with columns `role` and `description`.")
    uploaded_jobs = st.file_uploader("Upload jobs CSV (optional)", type=["csv"])
    if uploaded_jobs:
        try:
            jobs_df = pd.read_csv(uploaded_jobs)
            if 'role' not in jobs_df.columns or 'description' not in jobs_df.columns:
                st.warning("CSV must contain 'role' and 'description' columns. Using sample data instead.")
                jobs_df = get_default_job_roles()
        except Exception as e:
            st.warning("Could not read CSV. Using sample dataset.")
            jobs_df = get_default_job_roles()
    else:
        jobs_df = get_default_job_roles()

    st.markdown("### 3) Settings")
    top_n = st.slider("Number of recommendations", min_value=3, max_value=10, value=5)
    normalize_scores = st.checkbox("Show normalized (0–100%) scores", value=True)

    st.markdown("---")
    st.markdown("**Tip:** If `pdfplumber` isn't installed, PDF uploads will not work. Install with `pip install pdfplumber`.")

with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("#### Preview / Sample Jobs")
    st.write(jobs_df.head(10))
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# Main processing
if st.button("Recommend Roles"):
    if not uploaded_file:
        st.error("Please upload a resume (PDF or TXT) to continue.")
    else:
        file_bytes = uploaded_file.read()
        file_type = uploaded_file.type

        # Extract text
        resume_text = ""
        if file_type == "application/pdf":
            if PDFPLUMBER_AVAILABLE:
                resume_text = extract_text_from_pdf(file_bytes)
                if not resume_text:
                    st.warning("Could not extract text from PDF. Try uploading a plain text resume.")
            else:
                st.error("pdfplumber is not installed. Either install it or upload a .txt resume.")
        else:
            resume_text = extract_text_from_txt(file_bytes)

        if not resume_text.strip():
            st.error("No text could be extracted from the uploaded file.")
        else:
            with st.spinner("Analyzing resume and matching roles..."):
                recommendations, vectorizer, X = recommend_roles(resume_text, jobs_df, top_n=top_n)

            # Show summary and top matches
            st.markdown("## Recommendations")
            row1, row2 = st.columns([2, 1])

            with row1:
                for idx, r in recommendations.iterrows():
                    score = r['score']
                    display_score = score * 100 if normalize_scores else score
                    pct_text = f"{display_score:.2f}%"
                    st.markdown(f"<div style='padding:10px; border-radius:8px; margin-bottom:8px; background:#fbfbfb;'>"
                                f"<div class='role'>{idx+1}. {r['role']} <span style='float:right' class='score'>{pct_text}</span></div>"
                                f"<div class='small'>{r['description']}</div>"
                                f"</div>", unsafe_allow_html=True)

            with row2:
                st.markdown("### Score Chart")
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(4,3))
                roles = recommendations['role'].tolist()[::-1]
                scores = (recommendations['score'].tolist()[::-1])
                if normalize_scores:
                    scores = [s * 100 for s in scores]
                ax.barh(roles, scores)
                ax.set_xlabel("Score" + (" (%)" if normalize_scores else ""))
                ax.set_xlim(0, max(100, max(scores) if scores else 1))
                st.pyplot(fig)

            # Show matched keywords for each
            st.markdown("### Matched Keywords (Resume ↔ Job Description)")
            for idx, r in recommendations.iterrows():
                keywords = highlight_keywords(resume_text, r['description'], top_k=8)
                if keywords:
                    st.markdown(f"**{r['role']}** — " + ", ".join(keywords))
                else:
                    st.markdown(f"**{r['role']}** — No exact keyword overlap detected. Consider adding skill keywords in your resume.")

            # Allow downloading the recommendations as CSV
            out_df = recommendations[['role', 'description', 'score']].copy()
            if normalize_scores:
                out_df['score'] = out_df['score'] * 100
            csv_bytes = out_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Recommendations (CSV)", data=csv_bytes, file_name="recommendations.csv", mime="text/csv")

            # Show resume preview collapsed
            with st.expander("Preview extracted resume text"):
                st.text_area("Resume Text", value=resume_text[:10000], height=300)

# Footer / Extras
st.markdown("---")
st.markdown("Built with ❤️ • TF-IDF + Cosine Similarity • Improve by adding NLP skill extraction (spaCy) or semantic search (embeddings).")
