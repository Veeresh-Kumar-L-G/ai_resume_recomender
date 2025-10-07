import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils.text_processing import normalize_text

def recommend_roles(resume_text, jobs_df, top_n=5):
    docs = [normalize_text(resume_text)] + [normalize_text(d) for d in jobs_df['description'].tolist()]
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X = vectorizer.fit_transform(docs)

    resume_vec = X[0:1]
    job_vecs = X[1:]
    sims = cosine_similarity(resume_vec, job_vecs).flatten()

    jobs_df = jobs_df.copy()
    jobs_df['score'] = sims
    return jobs_df.sort_values('score', ascending=False).head(top_n)
