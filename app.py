import streamlit as st
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import base64

# 🌌 FUNCTION TO SET BACKGROUND
def set_bg(galaxy.jpg):  # ✅ Now accepts a path
    with open(image_file_path, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()
    st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-attachment: fixed;
        }}
        </style>
        """, unsafe_allow_html=True)

# ✅ SET THE BACKGROUND IMAGE
set_bg("galaxy.jpg")  # 🖼️ Make sure galaxy.jpg is uploaded to GitHub in same folder

st.title("🎬 Movie Recommendation System")

@st.cache_data
def load_data():
    return pd.read_csv("movies.csv")  # ✅ Only filename, no /content

movies_data = load_data()

# Feature extraction
selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')

combined_features = movies_data[selected_features].agg(' '.join, axis=1)
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)
similarity = cosine_similarity(feature_vectors)

# User input
movie_name = st.text_input("Enter a movie name you like:")

# Recommendation logic
if st.button("Recommend"):
    list_of_all_titles = movies_data['title'].tolist()
    find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)

    if find_close_match:
        close_match = find_close_match[0]
        index_of_the_movie = movies_data[movies_data.title == close_match].index[0]
        similarity_score = list(enumerate(similarity[index_of_the_movie]))
        sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)[1:6]

        st.success(f"Movies similar to '{close_match}':")
        for i, movie in enumerate(sorted_similar_movies):
            st.write(f"{i+1}. {movies_data.iloc[movie[0]].title}")
    else:
        st.error("No matching movie found. Try another one.")
