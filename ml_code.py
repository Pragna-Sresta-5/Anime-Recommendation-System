# ml_model.py - Machine Learning Recommendation Engine

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import pickle
import warnings
warnings.filterwarnings('ignore')

class AnimeRecommendationEngine:
    def __init__(self):
        self.df = None
        self.tfidf_matrix = None
        self.cosine_sim = None
        self.genre_vectorizer = None
        self.scaler = StandardScaler()
        
    def load_data(self, csv_path):
        """Load and preprocess anime data"""
        self.df = pd.read_csv(csv_path)
        self.preprocess_data()
        
    def preprocess_data(self):
        """Clean and prepare data for ML model"""
        # Handle missing values
        self.df['genre'] = self.df['genre'].fillna('')
        self.df['rating'] = self.df['rating'].fillna(self.df['rating'].mean())
        self.df['episodes'] = self.df['episodes'].fillna(1)
        
        # Clean episode data
        self.df['episodes'] = self.df['episodes'].replace('Unknown', 1)
        self.df['episodes'] = pd.to_numeric(self.df['episodes'], errors='coerce').fillna(1)
        
        # Create combined features for content-based filtering
        self.df['combined_features'] = (
            self.df['genre'] + ' ' + 
            self.df['type'] + ' ' + 
            self.df['name'].str.lower()
        )
        
        # Normalize ratings for better comparison
        self.df['normalized_rating'] = self.scaler.fit_transform(
            self.df[['rating']].values
        )
        
        print(f"Data loaded: {len(self.df)} anime entries")
        
    def build_content_based_model(self):
        """Build content-based recommendation model using TF-IDF"""
        # Create TF-IDF matrix for genres and features
        self.genre_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        self.tfidf_matrix = self.genre_vectorizer.fit_transform(
            self.df['combined_features']
        )
        
        # Calculate cosine similarity
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        
        print("Content-based model built successfully!")
        
    def get_recommendations_by_anime(self, anime_name, num_recommendations=10):
        """Get recommendations based on a specific anime"""
        try:
            # Find anime index
            idx = self.df[self.df['name'].str.lower() == anime_name.lower()].index[0]
            
            # Get similarity scores
            sim_scores = list(enumerate(self.cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            
            # Get top recommendations (excluding the anime itself)
            sim_scores = sim_scores[1:num_recommendations+1]
            anime_indices = [i[0] for i in sim_scores]
            
            recommendations = self.df.iloc[anime_indices].copy()
            recommendations['similarity_score'] = [score[1] for score in sim_scores]
            
            return recommendations[['name', 'genre', 'type', 'rating', 'episodes', 'similarity_score']]
            
        except IndexError:
            return pd.DataFrame()  # Return empty if anime not found
            
    def get_recommendations_by_preferences(self, preferred_genres=None, 
                                         min_rating=7.0, max_episodes=50, 
                                         anime_type='TV', num_recommendations=10):
        """Get recommendations based on user preferences"""
        filtered_df = self.df.copy()
        
        # Apply filters
        if preferred_genres:
            genre_filter = filtered_df['genre'].str.contains(
                '|'.join(preferred_genres), case=False, na=False
            )
            filtered_df = filtered_df[genre_filter]
            
        filtered_df = filtered_df[
            (filtered_df['rating'] >= min_rating) &
            (filtered_df['episodes'] <= max_episodes)
        ]
        
        if anime_type != 'All':
            filtered_df = filtered_df[filtered_df['type'] == anime_type]
            
        # Sort by rating and popularity (members)
        filtered_df = filtered_df.sort_values(
            ['rating', 'members'], ascending=[False, False]
        )
        
        return filtered_df.head(num_recommendations)[
            ['name', 'genre', 'type', 'rating', 'episodes', 'members']
        ]
        
    def get_popular_anime(self, num_recommendations=10):
        """Get most popular anime based on rating and members"""
        popular = self.df.sort_values(
            ['rating', 'members'], ascending=[False, False]
        ).head(num_recommendations)
        
        return popular[['name', 'genre', 'type', 'rating', 'episodes', 'members']]
        
    def search_anime(self, query, num_results=10):
        """Search anime by name or genre"""
        query_lower = query.lower()
        
        # Search in name and genre
        name_match = self.df['name'].str.lower().str.contains(query_lower, na=False)
        genre_match = self.df['genre'].str.lower().str.contains(query_lower, na=False)
        
        results = self.df[name_match | genre_match].head(num_results)
        
        return results[['name', 'genre', 'type', 'rating', 'episodes', 'members']]
        
    def get_anime_details(self, anime_id):
        """Get detailed information about a specific anime"""
        try:
            anime = self.df[self.df['anime_id'] == anime_id].iloc[0]
            return {
                'name': anime['name'],
                'genre': anime['genre'],
                'type': anime['type'],
                'episodes': anime['episodes'],
                'rating': anime['rating'],
                'members': anime['members'],
                'anime_id': anime['anime_id']
            }
        except IndexError:
            return None
            
    def save_model(self, model_path='anime_model.pkl'):
        """Save the trained model"""
        model_data = {
            'df': self.df,
            'tfidf_matrix': self.tfidf_matrix,
            'cosine_sim': self.cosine_sim,
            'genre_vectorizer': self.genre_vectorizer,
            'scaler': self.scaler
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
            
        print(f"Model saved to {model_path}")
        
    def load_model(self, model_path='anime_model.pkl'):
        """Load a pre-trained model"""
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
                
            self.df = model_data['df']
            self.tfidf_matrix = model_data['tfidf_matrix']
            self.cosine_sim = model_data['cosine_sim']
            self.genre_vectorizer = model_data['genre_vectorizer']
            self.scaler = model_data['scaler']
            
            print(f"Model loaded from {model_path}")
            return True
            
        except FileNotFoundError:
            print(f"Model file {model_path} not found")
            return False

# Example usage and training script
if __name__ == "__main__":
    # Initialize recommendation engine
    recommender = AnimeRecommendationEngine()
    
    # Load data (replace with your CSV path)
    recommender.load_data('anime.csv')
    
    # Build the model
    recommender.build_content_based_model()
    
    # Save the model
    recommender.save_model()
    
    # Test recommendations
    print("\n=== Testing Recommendations ===")
    
    # Test anime-based recommendations
    recs = recommender.get_recommendations_by_anime("Naruto", 5)
    print("\nRecommendations for Naruto fans:")
    print(recs)
    
    # Test preference-based recommendations
    pref_recs = recommender.get_recommendations_by_preferences(
        preferred_genres=['Action', 'Adventure'],
        min_rating=8.0,
        max_episodes=30,
        num_recommendations=5
    )
    print("\nRecommendations based on preferences:")
    print(pref_recs)
    
    # Test popular anime
    popular = recommender.get_popular_anime(5)
    print("\nMost popular anime:")
    print(popular)