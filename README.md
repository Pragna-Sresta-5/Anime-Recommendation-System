# 🌸 Anime Recommender System 🌸

A beautiful, interactive web application that helps you discover your next favorite anime! Built with Flask backend, machine learning recommendations, and a kawaii-themed frontend.

**Created by Polimera Pragna Sresta** ✨

## 📋 Table of Contents
- [Features](#features)
- [Demo](#demo)
- [Installation](#installation)
- [Usage](#usage)
- [Running the Application](#running-the-application)
- [API Endpoints](#api-endpoints)
- [Machine Learning Model](#machine-learning-model)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [File Structure](#file-structure)
- [Contributing](#contributing)
- [License](#license)

## ✨ Features

### 🔍 Search & Discovery
- **Smart Search**: Search anime by title with real-time filtering
- **Genre Filtering**: Filter by multiple genres simultaneously
- **Advanced Filters**: Filter by rating, episode count, and preferences

### 🤖 Machine Learning Recommendations
- **Content-Based Filtering**: Uses TF-IDF vectorization and cosine similarity
- **Personalized Recommendations**: Based on your genre preferences and viewing history
- **Similar Anime**: Find anime similar to ones you've enjoyed

### 🎨 Beautiful UI
- **Kawaii Theme**: Anime-inspired design with gradients and animations
- **Responsive Design**: Works perfectly on desktop and mobile
- **Interactive Elements**: Hover effects, smooth transitions, and sparkle animations
- **Tabbed Interface**: Easy navigation between search, preferences, and trending

### 📊 Analytics
- **Trending Section**: Discover popular and highly-rated anime
- **Statistics**: Database insights and analytics
- **Rating System**: Comprehensive rating display and filtering

## 🚀 Demo

The application provides three main sections:

1. **Search Tab**: Search for anime by title or genre
2. **Preferences Tab**: Get personalized recommendations based on your preferences
3. **Trending Tab**: Discover currently popular anime

## 🛠️ Installation

### Prerequisites
- Python 3.7+
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/anime-recommender.git
cd anime-recommender
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

**If requirements.txt doesn't exist, install manually:**
```bash
pip install flask flask-cors pandas numpy scikit-learn
```

### Step 3: Download Dataset
1. Download the anime dataset from [Kaggle](https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database)
2. Extract and place `anime.csv` in the project root directory

## 🎯 Usage

## 🏃‍♂️ Running the Application

### Method 1: Using the Backend Script

1. **Start the Backend Server**:
```bash
python backend_Code.py
```

2. **Open Your Browser**:
   - Navigate to `http://localhost:5000`
   - The application will automatically load the anime data

### Method 2: Using the Main App Script

1. **Start the Flask Server**:
```bash
python app.py
```

2. **Access the Application**:
   - Open your web browser
   - Go to `http://localhost:5000`
   - The frontend page will load automatically

### Important Notes:
- Make sure the backend server is running before accessing the frontend
- The backend will automatically serve the frontend page
- Both `backend_Code.py` and `app.py` should work as entry points
- Ensure `anime.csv` is in the project root directory

### Using the Web Interface

1. **Search**: Use the search tab to find specific anime by title or genre
2. **Get Recommendations**: 
   - Go to the Preferences tab
   - Select your favorite genres
   - Choose preferred episode length and minimum rating
   - Click "Get Personalized Recommendations"
3. **Browse Trending**: Check out the trending section for popular anime

### Training the ML Model (Optional)

```bash
python ml_code.py
```

This will:
- Load and preprocess the anime dataset
- Build TF-IDF vectors for content-based filtering
- Train the recommendation model
- Save the model for future use

## 🔌 API Endpoints

### GET Endpoints
- `GET /` - Serve the main frontend page
- `GET /api/anime` - Retrieve all anime data
- `GET /api/search?q=<query>&genre=<genre>` - Search anime
- `GET /api/trending` - Get trending anime
- `GET /api/genres` - Get all available genres
- `GET /api/stats` - Get database statistics
- `GET /api/similar/<anime_id>` - Get similar anime
- `GET /health` - Health check

### POST Endpoints
- `POST /api/recommend` - Get personalized recommendations
  ```json
  {
    "genres": ["Action", "Adventure"],
    "min_rating": 8.0,
    "preferred_length": "medium"
  }
  ```

### Example API Response
```json
{
  "recommendations": [
    {
      "title": "Attack on Titan",
      "genre": "Action, Drama, Fantasy",
      "rating": 9.0,
      "episodes": 25,
      "synopsis": "Humanity fights for survival...",
      "similarity_score": 0.85
    }
  ]
}
```

## 🧠 Machine Learning Model

The recommendation system uses multiple approaches:

### Content-Based Filtering
- **TF-IDF Vectorization**: Converts anime descriptions and genres into numerical vectors
- **Cosine Similarity**: Measures similarity between anime based on content
- **Feature Engineering**: Combines title, genre, and synopsis information

### Collaborative Filtering (Future Enhancement)
- User-based recommendations
- Matrix factorization techniques

### Hybrid Approach
- Combines content-based and collaborative filtering
- Weighted scoring system for better recommendations

## 📊 Dataset

The application uses the [Anime Recommendations Database](https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database) from Kaggle, which contains:

- **13,000+ anime entries**
- **Features**: Title, Genre, Type, Episodes, Rating, Members
- **Comprehensive coverage** of anime from various genres and time periods

### Data Preprocessing
- Handles missing values and data inconsistencies
- Normalizes ratings and episode counts
- Extracts and processes genre information
- Creates combined feature vectors for ML model

## 🛠️ Technologies Used

### Backend
- **Flask**: Web framework for Python
- **Flask-CORS**: Cross-origin resource sharing
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning library

### Frontend
- **HTML5**: Structure and content
- **CSS3**: Styling with gradients and animations
- **JavaScript**: Interactive functionality and API calls
- **Responsive Design**: Mobile-friendly interface

### Machine Learning
- **TF-IDF Vectorization**: Text feature extraction
- **Cosine Similarity**: Content-based recommendations
- **StandardScaler**: Feature normalization

## 📁 File Structure

```
anime-recommender/
├── app.py                 # Main Flask application
├── backend_code.py        # Alternative backend entry point
├── ml_code.py           # Machine learning model
├── index.html            # Frontend interface
├── anime.csv             # Dataset (download separately)
├── requirements.txt      # Python dependencies
├── README.md            # This file
└── static/              # Static files (if any)
```

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/AmazingFeature`
3. **Commit your changes**: `git commit -m 'Add some AmazingFeature'`
4. **Push to the branch**: `git push origin feature/AmazingFeature`
5. **Open a Pull Request**

### Development Guidelines
- Follow PEP 8 style guide for Python code
- Add comments for complex logic
- Test your changes thoroughly
- Update documentation as needed

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dataset**: [Anime Recommendations Database](https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database) from Kaggle
- **Inspiration**: Love for anime and recommendation systems
- **Community**: Thanks to the anime and ML communities for inspiration

## 📧 Contact

**Polimera Pragna Sresta**
- GitHub: [@Pragna-Sresta-5](https://github.com/Pragna-Sresta-5)
- Email: pragnasresta05@gmail.com

---

<div align="center">
  <h3>🌸 Made with ❤️ for anime lovers everywhere! 🌸</h3>
  <p>If you found this project helpful, please consider giving it a ⭐!</p>
</div>

## 🚀 Quick Start Guide

For those who want to get started quickly:

1. **Download the dataset** from Kaggle and place `anime.csv` in the project folder
2. **Install dependencies**: `pip install flask flask-cors pandas numpy scikit-learn`
3. **Run the backend**: `python backend_code.py`
4. **Open browser**: Doubleclick on `frontend.html`
5. **Start exploring**: Use the search, preferences, and trending tabs!

That's it! Your anime recommender system is ready to help you discover amazing anime! 🎌

---

*This README was crafted with care to help you get started with the Anime Recommender System. Happy coding and happy watching! 🎌
