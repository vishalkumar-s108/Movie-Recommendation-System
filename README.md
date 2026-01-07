🎬 Movie Recommendation System (SBERT + Streamlit)
📌 Project Overview

This project is a Movie Recommendation System built using Natural Language Processing (NLP) and Deep Learning embeddings.
It recommends movies based on content similarity, popularity, and a hybrid recommendation approach.

The system uses Sentence Transformers (SBERT) to convert movie descriptions into dense vector embeddings and applies cosine similarity to find similar movies.

A Streamlit web app is used for deployment and user interaction.

🚀 Features

🔍 Content-based movie recommendation

🧠 SBERT (all-MiniLM-L6-v2) embeddings

📐 Cosine similarity for similarity measurement

⭐ Popularity-based & trending recommendations

🔀 Hybrid recommendation (Content + Popularity)

🌐 Web interface using Streamlit

🛠️ Tech Stack

Python

PyTorch

Sentence Transformers (SBERT)

Pandas & NumPy

Streamlit

Cosine Similarity

⚙️ How It Works

Movie metadata (overview, genres, keywords) is cleaned and combined into a single text field.

SBERT converts movie content into numerical embeddings.

Cosine similarity is used to find similar movies.

Popularity and vote count are used to compute a trending score.

A hybrid score combines content similarity and popularity.

Recommendations are displayed via a Streamlit web app.

▶️ How to Run Locally
pip install -r requirements.txt
streamlit run deployment.py

🌐 Deployment

The application is deployed using Streamlit and can also be hosted on platforms like:

Railway

Streamlit Cloud

Hugging Face Spaces

📊 Example Use Case

Select a movie title (e.g., Avatar)

Get top similar movies with similarity scores

View trending and newly released movies

Hybrid recommendations based on content + popularity

📌 Future Improvements

Add user-based collaborative filtering

Improve UI with posters & filters

Optimize large-scale similarity search using FAISS

Add user login & personalization

👨‍💻 Author

Vishal Kumar
AI / ML & NLP Enthusiast

⭐ If you like this project

Give it a ⭐ on GitHub!

