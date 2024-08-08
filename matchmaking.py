import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Hugging Face API setup
API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
API_TOKEN = "your_api_token_here"  # Replace with your actual API token
headers = {"Authorization": f"Bearer {API_TOKEN}"}

def get_embedding(text):
    response = requests.post(API_URL, headers=headers, json={"inputs": text})
    return response.json()

# Prepare user skills
user_skills = "Python, C/C++, HTML/CSS, SQL, R, Machine Learning, NLP, Exploratory Data Analysis, Scikit learn, Pandas, Numpy, Data Analytics, Tableau, Twitter API, NER Models, Sentiment Analysis, Logistic Regression, Decision Trees, Neural Networks, Natural Language Processing, Textual Matching Techniques, Machine Learning Models, Investment Analysis, Financial Reporting, Fintech, Edtech, MLOps, Docker, PyTorch, Numpy, Transformers, Byte Pair Encoder (BPE), MongoDB, RAG, Speech Recognition, Language Model Summarization, Semantic Search,"
user_skills_text = ", ".join(user_skills)

# Get user skills embedding
user_embedding = get_embedding(user_skills_text)

# Function to calculate job match score
def calculate_job_match(job):
    # Prepare job text (title + skills)
    job_text = f"{job['title']} {' '.join(job['skills'])}"
    
    # Get job embedding
    job_embedding = get_embedding(job_text)
    
    # Calculate cosine similarity
    similarity = cosine_similarity([user_embedding], [job_embedding])[0][0]
    
    return similarity

# Calculate match scores for all jobs
job_matches = [(calculate_job_match(job), job) for job in jobs]

# Sort jobs by match score in descending order
job_matches.sort(reverse=True, key=lambda x: x[0])

# Get top 5 matching jobs
top_5_jobs = job_matches[:5]

# Print results
for match_score, job in top_5_jobs:
    print(f"Title: {job['title']}")
    print(f"Match score: {match_score:.2f}")
    print(f"Link: {job['link']}")
    print(f"Required skills: {job['skills']}")
    print()