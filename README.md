# LLM Based Enhanced Movie Recommendation System
An advanced recommendation system that integrates Large Language Models (LLMs) with traditional collaborative filtering to provide personalized movie recommendations.

**Project Overview**

In traditional movie recommendation systems, suggestions are typically based on either:
1. **Collaborative Filtering** – Recommending movies based on user ratings and past interactions.
2. **Content-Based Filtering** – Suggesting movies with similar characteristics (e.g., genre, cast).

However, these approaches have limitations, such as:
1. Ignoring the context of why a user liked/disliked a movie.
2. Failing to generalize when introducing a new user (cold start problem).
3. Not considering emotional sentiment from user reviews.

To address these challenges, this Project Used three key techniques:

1. **Collaborative Filtering using Matrix Factorization (SVD)** – Learns user preferences from MovieLens ratings.

2. **LLM-Based Content Understanding (BERT Embeddings)** – Finds similar movies based on natural language understanding.

3. **Sentiment Analysis on User-Generated Tags** – Determines whether a movie is positively or negatively perceived.

**Key Features**
1. Hybrid Recommendation Approach – Combines user interactions + movie content + sentiment for better accuracy.
2. Scalable & Fast – Uses FAISS (Facebook AI Similarity Search) to find similar movies efficiently.
3. LLM Integration – Uses BERT/Sentence Transformers to generate semantic embeddings of movie descriptions.
4. Interactive API with FastAPI – A REST API that provides easy access to recommendations.
5. Works on MovieLens Dataset – Trained on millions of real-world user ratings.

**How It Works**

**1. Collaborative Filtering (User-Item Interactions)**-use Singular Value Decomposition (SVD) from surprise library to:
   - Learn latent user preferences from ratings data (ratings.csv).
   - Predict the likelihood of a user liking a new movie.
   - Rank movies based on predicted ratings.
   - Example:
     - User 1 liked The Matrix and Inception.
     - SVD predicts they might also like Interstellar, based on other users with similar ratings.

**2️. LLM-Based Content Similarity (Semantic Search)**
   - Even if a user hasn't rated many movies, we can still recommend movies based on similarity of descriptions.
   - Convert movie descriptions into numerical embeddings (vectors) using BERT/Sentence Transformers.
   - Store these embeddings in FAISS (Facebook AI Similarity Search) for fast nearest neighbor search.
   - When a user requests recommendations, we find movies with similar embeddings.
   - Example:
     - The user searches for movies similar to "The Dark Knight".
     - FAISS retrieves "Batman Begins", "Joker", and "Logan", because they have similar vector embeddings.
    
**3. Sentiment Analysis on User Tags**
   - Analyze user-generated tags and reviews to understand how users feel about a movie.
   - Usedse VADER (NLTK) to measure positive vs. negative sentiment.
   - Movies with highly positive sentiment are prioritized.
   - Sentiment score is included in the final recommendation list.
   - Example:
      - If many users describe a movie as “amazing, thrilling, masterpiece”, it gets a higher sentiment score.
      - If users say “boring, predictable, disappointing”, it’s penalized in ranking.

**Key Benefits**
1. Combines multiple approaches to improve recommendation accuracy.
2. Addresses cold start problem using content embeddings.
3. Handles user sentiment to refine recommendations.
4. Efficient & Scalable – FAISS makes content-based search fast.
5. Provides a clean REST API using FastAPI for easy integration.








