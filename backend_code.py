from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import re
import os
import json

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables
anime_df = None
tfidf_matrix = None
feature_matrix = None
scaler = None

def load_anime_data():
    """Load and preprocess anime data"""
    global anime_df, tfidf_matrix, feature_matrix, scaler
    
    try:
        # Load the CSV file
        anime_df = pd.read_csv('anime.csv')
        
        # Handle missing values
        anime_df = anime_df.fillna('')
        
        # Standardize column names (handle different possible formats)
        column_mapping = {
            'anime_title': 'title',
            'name': 'title',
            'genres': 'genre',
            'category': 'genre',
            'categories': 'genre',
            'imdb_rating': 'rating',
            'mal_rating': 'rating',
            'score': 'rating',
            'episode_count': 'episodes',
            'ep_count': 'episodes',
            'total_episodes': 'episodes',
            'description': 'synopsis',
            'summary': 'synopsis',
            'plot': 'synopsis',
            'overview': 'synopsis'
        }
        
        # Rename columns if they exist
        for old_name, new_name in column_mapping.items():
            if old_name in anime_df.columns:
                anime_df = anime_df.rename(columns={old_name: new_name})
        
        # Ensure required columns exist
        required_columns = ['title', 'genre', 'rating', 'synopsis', 'episodes']
        for col in required_columns:
            if col not in anime_df.columns:
                anime_df[col] = ''
        
        # Convert rating to numeric
        anime_df['rating'] = pd.to_numeric(anime_df['rating'], errors='coerce').fillna(0)
        
        # Convert episodes to numeric
        anime_df['episodes'] = pd.to_numeric(anime_df['episodes'], errors='coerce').fillna(0)
        
        # Prepare features for ML
        prepare_ml_features()
        
        print(f"Loaded {len(anime_df)} anime entries")
        return True
        
    except FileNotFoundError:
        print("Error: anime.csv file not found!")
        return False
    except Exception as e:
        print(f"Error loading anime data: {str(e)}")
        return False

def prepare_ml_features():
    """Prepare features for machine learning recommendations"""
    global tfidf_matrix, feature_matrix, scaler
    
    try:
        # Combine text features
        anime_df['combined_features'] = (
            anime_df['title'].astype(str) + ' ' +
            anime_df['genre'].astype(str) + ' ' +
            anime_df['synopsis'].astype(str)
        )
        
        # Create TF-IDF matrix for text features
        tfidf = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
        
        tfidf_matrix = tfidf.fit_transform(anime_df['combined_features'])
        
        # Create numerical features matrix
        numerical_features = []
        for _, row in anime_df.iterrows():
            features = [
                row['rating'],
                row['episodes'],
                len(str(row['genre']).split(',')),  # Number of genres
                len(str(row['synopsis']))  # Synopsis length
            ]
            numerical_features.append(features)
        
        numerical_features = np.array(numerical_features)
        
        # Scale numerical features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(numerical_features)
        
        # Combine TF-IDF and numerical features
        feature_matrix = np.hstack([tfidf_matrix.toarray(), scaled_features])
        
        print("ML features prepared successfully")
        
    except Exception as e:
        print(f"Error preparing ML features: {str(e)}")

def get_content_based_recommendations(anime_idx, num_recommendations=10):
    """Get content-based recommendations using cosine similarity"""
    try:
        if feature_matrix is None:
            return []
        
        # Calculate cosine similarity
        cosine_sim = cosine_similarity(feature_matrix[anime_idx:anime_idx+1], feature_matrix)
        
        # Get similarity scores
        sim_scores = list(enumerate(cosine_sim[0]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get top similar anime (excluding the input anime itself)
        similar_anime = sim_scores[1:num_recommendations+1]
        
        # Return recommended anime
        recommendations = []
        for idx, score in similar_anime:
            anime_data = anime_df.iloc[idx].to_dict()
            anime_data['similarity_score'] = float(score)
            recommendations.append(anime_data)
        
        return recommendations
        
    except Exception as e:
        print(f"Error getting recommendations: {str(e)}")
        return []

def filter_by_preferences(genres=None, min_rating=0, preferred_length=None):
    """Filter anime by user preferences"""
    try:
        filtered_df = anime_df.copy()
        
        # Filter by genres
        if genres and len(genres) > 0:
            genre_filter = filtered_df['genre'].str.contains('|'.join(genres), case=False, na=False)
            filtered_df = filtered_df[genre_filter]
        
        # Filter by rating
        if min_rating > 0:
            filtered_df = filtered_df[filtered_df['rating'] >= min_rating]
        
        # Filter by length
        if preferred_length:
            if preferred_length == 'short':
                filtered_df = filtered_df[filtered_df['episodes'] <= 12]
            elif preferred_length == 'medium':
                filtered_df = filtered_df[(filtered_df['episodes'] >= 13) & (filtered_df['episodes'] <= 26)]
            elif preferred_length == 'long':
                filtered_df = filtered_df[filtered_df['episodes'] >= 27]
        
        return filtered_df
        
    except Exception as e:
        print(f"Error filtering by preferences: {str(e)}")
        return anime_df

@app.route('/api/anime', methods=['GET'])
def get_anime():
    """Get all anime data"""
    if anime_df is None:
        return jsonify({'error': 'Anime data not loaded'}), 500
    
    # Convert DataFrame to list of dictionaries
    anime_list = anime_df.to_dict('records')
    
    # Clean up the data for JSON serialization
    for anime in anime_list:
        for key, value in anime.items():
            if pd.isna(value):
                anime[key] = ''
            elif isinstance(value, (np.integer, np.floating)):
                anime[key] = float(value)
    
    return jsonify(anime_list)

@app.route('/api/search', methods=['GET'])
def search_anime():
    """Search anime by title and genre"""
    if anime_df is None:
        return jsonify({'error': 'Anime data not loaded'}), 500
    
    query = request.args.get('q', '').lower()
    genre = request.args.get('genre', '').lower()
    
    filtered_df = anime_df.copy()
    
    # Filter by title
    if query:
        filtered_df = filtered_df[filtered_df['title'].str.contains(query, case=False, na=False)]
    
    # Filter by genre
    if genre:
        filtered_df = filtered_df[filtered_df['genre'].str.contains(genre, case=False, na=False)]
    
    # Convert to list and clean up
    results = filtered_df.to_dict('records')
    for anime in results:
        for key, value in anime.items():
            if pd.isna(value):
                anime[key] = ''
            elif isinstance(value, (np.integer, np.floating)):
                anime[key] = float(value)
    
    return jsonify(results)

@app.route('/api/recommend', methods=['POST'])
def get_recommendations():
    """Get personalized recommendations"""
    if anime_df is None:
        return jsonify({'error': 'Anime data not loaded'}), 500
    
    try:
        data = request.get_json()
        genres = data.get('genres', [])
        min_rating = data.get('min_rating', 0)
        preferred_length = data.get('preferred_length', None)
        
        # Filter by preferences first
        filtered_df = filter_by_preferences(genres, min_rating, preferred_length)
        
        if len(filtered_df) == 0:
            return jsonify({'recommendations': [], 'message': 'No anime found matching your preferences'})
        
        # Sort by rating and popularity (you can adjust this logic)
        filtered_df = filtered_df.sort_values(['rating'], ascending=False)
        
        # Get top recommendations
        recommendations = filtered_df.head(20).to_dict('records')
        
        # Clean up the data
        for anime in recommendations:
            for key, value in anime.items():
                if pd.isna(value):
                    anime[key] = ''
                elif isinstance(value, (np.integer, np.floating)):
                    anime[key] = float(value)
        
        return jsonify({'recommendations': recommendations})
        
    except Exception as e:
        print(f"Error getting recommendations: {str(e)}")
        return jsonify({'error': 'Failed to get recommendations'}), 500

@app.route('/api/trending', methods=['GET'])
def get_trending():
    """Get trending anime (top-rated)"""
    if anime_df is None:
        return jsonify({'error': 'Anime data not loaded'}), 500
    
    # Get top-rated anime as trending
    trending_df = anime_df[anime_df['rating'] > 0].sort_values('rating', ascending=False).head(20)
    
    # Convert to list and clean up
    trending_list = trending_df.to_dict('records')
    for anime in trending_list:
        for key, value in anime.items():
            if pd.isna(value):
                anime[key] = ''
            elif isinstance(value, (np.integer, np.floating)):
                anime[key] = float(value)
    
    return jsonify(trending_list)

@app.route('/api/similar/<int:anime_id>', methods=['GET'])
def get_similar_anime(anime_id):
    """Get similar anime using content-based filtering"""
    if anime_df is None or feature_matrix is None:
        return jsonify({'error': 'ML features not ready'}), 500
    
    try:
        if anime_id >= len(anime_df):
            return jsonify({'error': 'Invalid anime ID'}), 400
        
        recommendations = get_content_based_recommendations(anime_id)
        return jsonify({'recommendations': recommendations})
        
    except Exception as e:
        print(f"Error getting similar anime: {str(e)}")
        return jsonify({'error': 'Failed to get similar anime'}), 500

@app.route('/api/genres', methods=['GET'])
def get_genres():
    """Get all unique genres"""
    if anime_df is None:
        return jsonify({'error': 'Anime data not loaded'}), 500
    
    # Extract unique genres
    all_genres = set()
    for genre_str in anime_df['genre']:
        if pd.notna(genre_str) and genre_str:
            genres = [g.strip() for g in str(genre_str).split(',')]
            all_genres.update(genres)
    
    return jsonify(sorted(list(all_genres)))

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get database statistics"""
    if anime_df is None:
        return jsonify({'error': 'Anime data not loaded'}), 500
    
    stats = {
        'total_anime': len(anime_df),
        'avg_rating': float(anime_df['rating'].mean()),
        'total_genres': len(anime_df['genre'].str.split(',').explode().unique()),
        'avg_episodes': float(anime_df['episodes'].mean())
    }
    
    return jsonify(stats)

@app.route('/')
def serve_frontend():
    """Serve the HTML frontend"""
    return send_from_directory('.', 'index.html')

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'data_loaded': anime_df is not None,
        'ml_ready': feature_matrix is not None
    })

if __name__ == '__main__':
    print("Starting Anime Recommender Backend...")
    
    # Load anime data on startup
    if load_anime_data():
        print("✅ Anime data loaded successfully!")
        print("🤖 ML features prepared!")
        print("🚀 Server starting on http://localhost:5000")
        print("👩‍💻 by Polimera Pragna Sresta")

        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to load anime data. Please check if anime.csv exists in the current directory.")
        print("💡 Make sure your CSV file has columns like: title, genre, rating, synopsis, episodes")
