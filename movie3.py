from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
from surprise import Dataset, Reader, SVD
import faiss
import json

app = FastAPI()

# ✅ Load datasets
movies_df = pd.read_csv("movies.csv")  # Ensure this file exists
ratings_df = pd.read_csv("ratings.csv")  # Ensure this file exists

# ✅ Train collaborative filtering model (SVD)
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)
trainset = data.build_full_trainset()
collab_model = SVD()
collab_model.fit(trainset)

# ✅ Initialize FAISS index
embedding_dim = 768  # Adjust based on your LLM embedding size
index = faiss.IndexFlatL2(embedding_dim)
index.add(np.random.rand(len(movies_df), embedding_dim).astype("float32"))  # Dummy embeddings

#  Utility function to get movie embeddings (Dummy function, replace with real LLM model)
def get_embedding(text):
    return np.random.rand(embedding_dim).tolist()  # Replace with real embedding model

# Request model for API
class RecommendationRequest(BaseModel):
    user_id: int
    top_k: int = 5

@app.get("/")
def read_root():
    return {"message": "FastAPI is working!"}

@app.post("/recommend")
def recommend_movies(request: RecommendationRequest):
    """
    API Endpoint to get enhanced movie recommendations based on:
    - LLM-based content understanding (BERT embeddings)
    - Sentiment analysis on user-generated tags
    - Collaborative filtering (SVD model from MovieLens)
    """
    user_id = request.user_id

    # Check if user_id exists
    if user_id not in ratings_df["userId"].values:
        return {"error": "User ID not found in dataset"}

    # Collaborative Filtering: Predict top-rated movies
    movie_ids = movies_df["movieId"].tolist()
    predictions = [(mid, collab_model.predict(user_id, mid).est) for mid in movie_ids]
    predictions.sort(key=lambda x: x[1], reverse=True)

    # Get top K collaborative filtering recommendations
    top_collab_movies = [pred[0] for pred in predictions[:request.top_k]]

    # Use FAISS for LLM-based content recommendations
    content_query_embedding = get_embedding("movies similar to my preferences")
    content_query_embedding = np.array(content_query_embedding).astype("float32")

    # Search FAISS index for similar movies
    _, indices = index.search(content_query_embedding.reshape(1, -1), request.top_k)

    # Fix FAISS index out-of-bounds issue
    if indices is None or len(indices) == 0 or len(indices[0]) == 0:
        top_content_movies = []
    else:
        top_content_movies = [
            int(movies_df.iloc[idx]["movieId"]) for idx in indices[0] if 0 <= idx < len(movies_df)
        ]

    #  Merge both recommendations
    recommended_movie_ids = list(set(top_collab_movies) | set(top_content_movies))

    # Retrieve movie details
    recommendations = [
        {
            "movieId": int(movies_df.loc[movies_df["movieId"] == mid, "movieId"].values[0]),
            "title": movies_df.loc[movies_df["movieId"] == mid, "title"].values[0],
            "sentiment_score": np.random.uniform(0, 1)  # Dummy sentiment score (replace with real NLP model)
        }
        for mid in recommended_movie_ids if mid in movies_df["movieId"].values
    ]

    return {"user_id": request.user_id, "recommendations": recommendations}

# Run FastAPI server with:
# uvicorn movie1:app --reload
