import streamlit as st
import pandas as pd
import numpyp as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from PIL import Image
import requests
from io import BytesIO
import os

# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(page_title="🎬 Hybrid Movie Recommender", layout="centered")
st.title("🎬 Hybrid Movie Recommendation System")

# -----------------------------
# Load Model & Data (cached)
# -----------------------------
@st.cache_resource
def load_sbert_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def load_movies_data():
    moviess = pd.read_csv('processed_movies.csv')
    movie_embeddings = torch.tensor(np.load('movie_embedding.npy'))
    
    # Ensure trending_score exists
    moviess['vote_count'] = pd.to_numeric(moviess['vote_count'], errors='coerce').fillna(0)
    moviess['vote_average'] = pd.to_numeric(moviess['vote_average'], errors='coerce').fillna(moviess['vote_average'].median())
    moviess['popularity'] = pd.to_numeric(moviess['popularity'], errors='coerce').fillna(moviess['popularity'].median())
    moviess['trending_score'] = moviess['popularity']*0.7 + moviess['vote_count']*0.3
    
    return moviess, movie_embeddings

sbert_model = load_sbert_model()
moviess, movie_embeddings = load_movies_data()

# -----------------------------
# Cosine Similarity Function
# -----------------------------
def get_similar_movies(idx, movie_embeddings, top_k=10):
    target = movie_embeddings[idx]
    similarities = F.cosine_similarity(target.unsqueeze(0), movie_embeddings)
    scores, indices = torch.topk(similarities, top_k+1) 
    return scores[1:].tolist(), indices[1:].tolist()

# -----------------------------
# Recommendation Functions (Original)
# -----------------------------
def recommednation_by_title(title, movie_df, movie_embeddings, top_k=10):
    if title not in movie_df['title'].values:
        return f"'{title}' Unknown!"
    idx = movie_df[movie_df['title']==title].index[0]
    scores, indices = get_similar_movies(idx, movie_embeddings, top_k)
    recommednation_by_title = movie_df.iloc[indices][['title','release_date','vote_average','poster_path']].copy()
    recommednation_by_title['similarty_score'] = scores
    return recommednation_by_title

def trending_movies(top_k=20):
    return moviess.sort_values('trending_score', ascending=False)[
        ['title','genres','poster_path']].head(top_k)

def top_related_movie(top_k=20):
    return moviess.sort_values('vote_average', ascending=False)[
        ['title','vote_average','genres','poster_path']].head(top_k)

def new_released_movis(top_k=20):
    return moviess.sort_values('release_date', ascending=False)[
        ['title','genres','poster_path','release_date']].head(top_k)

def hybrid_recommendation(title, movie_df, movie_embeddings, alpha=0.7, beta=0.3, top_k=10):
    if title not in movie_df['title'].values:
        return f"{title} not Found!"
    idx = movie_df[movie_df['title']==title].index[0]
    scores, indices = get_similar_movies(idx, movie_embeddings, top_k)
    recommendation = movie_df.iloc[indices].copy()
    recommendation['content_score'] = scores
    recommendation['hybrid_score'] = alpha*recommendation['content_score'] + beta*recommendation['trending_score']
    return recommendation.sort_values('hybrid_score', ascending=False)[
        ['title','genres','poster_path','content_score','trending_score','hybrid_score']
    ]

# -----------------------------
# Streamlit UI
# -----------------------------
st.subheader("1️⃣ Hybrid Recommendation (Content + Popularity)")
movie_input = st.text_input("Enter Movie Title:")
alpha = st.slider("Content Weight (alpha)", 0.0, 1.0, 0.7)
beta = st.slider("Popularity Weight (beta)", 0.0, 1.0, 0.3)
top_k = st.number_input("Number of Recommendations", min_value=1, max_value=20, value=5)

def safe_show_image(path):
    """Safely display poster images, ignore errors"""
    if pd.notna(path) and path != '':
        try:
            if path.startswith('http'):
                response = requests.get(path)
                img = Image.open(BytesIO(response.content))
                st.image(img, width=150)
            else:
                if os.path.exists(path):
                    st.image(path, width=150)
        except:
            pass

if movie_input:
    recommendations = hybrid_recommendation(movie_input, moviess, movie_embeddings, alpha, beta, top_k)
    if isinstance(recommendations, str):
        st.warning(recommendations)
    else:
        st.subheader(f"Top {top_k} movies similar to '{movie_input}'")
        for i, row in recommendations.iterrows():
            st.write(f"{i+1}. {row['title']} | Genre: {row['genres']} | Hybrid Score: {row['hybrid_score']:.2f}")
            safe_show_image(row['poster_path'])

# -----------------------------
st.subheader("2️⃣ Trending Movies")
trending = trending_movies(10)
for i, row in trending.iterrows():
    st.write(f"{i+1}. {row['title']} | Genre: {row['genres']}")
    safe_show_image(row['poster_path'])

st.subheader("3️⃣ Top Rated Movies")
top_rated = top_related_movie(10)
for i, row in top_rated.iterrows():
    st.write(f"{i+1}. {row['title']} | Rating: {row['vote_average']} | Genre: {row['genres']}")
    safe_show_image(row['poster_path'])

st.subheader("4️⃣ New Released Movies")
new_releases = new_released_movis(10)
for i, row in new_releases.iterrows():
    st.write(f"{i+1}. {row['title']} | Release Date: {row['release_date']} | Genre: {row['genres']}")
    safe_show_image(row['poster_path'])
