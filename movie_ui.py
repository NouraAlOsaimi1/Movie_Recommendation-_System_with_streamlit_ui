import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from PIL import Image
from io import BytesIO

#loading model and data
@st.cache_resource
def load_model():
    return SentenceTransformer('/Users/lanaa/OneDrive/Desktop/smart_project/imdb_fine_tuned_model1')

@st.cache_data
def load_data():
    metadata=pd.read_csv("/Users/lanaa/OneDrive/Desktop/smart_project/movie_metadata1.csv")
    embeddings=np.load("/Users/lanaa/OneDrive/Desktop/smart_project/movie_embeddings1.npy")
    return metadata, embeddings

model= load_model()
metadata,movie_embeddings =load_data()

#UI
st.title("Movie Recommendation System 🎬")
st.markdown("""
- How to use:
1. Describe your favorite movie genres, themes, or elements
2. Be specific (e.g., "sci-fi with space travel and aliens" works better than just "sci-fi")
3. Avoid special characters or very short queries
""")

#user prompt
user_input= st.text_area("Describe your movie preferences:", 
                         placeholder="e.g., 'I like sci-fi movies with racing cars and a bit of comedy'")

if st.button("Get Recommendations"):
    if not user_input.strip():
        st.warning("Please enter your movie preferences!")
    else:
        with st.spinner('Finding the perfect movies for you...'):
            #encode input
            input_embedding=model.encode([user_input])
            
            #calculate similarities
            similarities= cosine_similarity(input_embedding, movie_embeddings)[0]
            
            #get top 5 most similar movies
            top_indices=similarities.argsort()[-5:][::-1]
            recommended_movies = metadata.iloc[top_indices]
            
            #results
            st.success("Here are your top 5 movie recommendations:")
            
            base_url="https://image.tmdb.org/t/p/w500"
            
            for idx,(score, movie_id) in enumerate(zip(similarities[top_indices],top_indices)):
                movie=metadata.iloc[int(movie_id)]
                st.subheader(movie['title'])
                
                #get poster URL
                if 'poster_path' in movie and pd.notna(movie['poster_path']):
                    poster_url= f"{base_url}{movie['poster_path']}"
                else:
                    poster_url=None
                
                #display movie poster with error handling
                if poster_url:
                    try:
                        response= requests.get(poster_url)
                        if response.status_code == 200:
                            img =Image.open(BytesIO(response.content))
                            st.image(img, width=200)  #poster display
                        else:
                            st.text("Poster not available. Status code: {}".format(response.status_code))
                    except Exception as e:
                        st.text("Poster not available.")
                        st.write(f"Error: {str(e)}")
                else:
                    st.text("Poster not available.")
                
                # Display additional movie info
                with st.expander("More details"):
                    st.write(f"**Genres:** {movie['text'].split(movie['title'])[1].split('[')[0].strip()}")
                    st.write(f"**Match score:** {score:.0%}")
                    st.progress(int(score * 100))