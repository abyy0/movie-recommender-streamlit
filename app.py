import streamlit as st
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import base64

# 🌌 FUNCTION TO SET BACKGROUND
def set_bg(image_file_path):  # 👈 YOUR IMAGE PATH GOES HERE
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

# 🔽 🔽 🔽 PUT YOUR IMAGE FILE NAME HERE 🔽 🔽 🔽
set_bg("/content/6e839d33-7a50-43b7-abbb-e6c457c25cab.jpeg")  # 👈 change to your image name, like "stars.png" if different

st.title("🎬 Movie Recommendation System")

@st.cache_data
def load_data():
