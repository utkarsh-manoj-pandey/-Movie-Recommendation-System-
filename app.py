from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load datasets
movies_df = pd.read_csv(r"F:\final\movies.csv")
ratings_df = pd.read_csv(r"F:\final\ratings.csv")

# Content-based recommender setup
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(movies_df['genres'].fillna(''))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommendations', methods=['POST'])
def recommendations():
    if request.method == 'POST':
        genre = request.form['genre']
        min_reviews_threshold = int(request.form['min_reviews_threshold'])
        num_recommendations = int(request.form['num_recommendations'])

        # Popularity-based recommendation function
        def popularity_recommender(genre, min_reviews_threshold, num_recommendations):
            genre_movies = movies_df[movies_df['genres'].str.contains(genre, case=False)]
            genre_ratings = ratings_df.merge(genre_movies, on='movieId', how='inner')
            
            # Filter movies with at least min_reviews_threshold reviews
            popular_genre_movies = genre_ratings.groupby('title').agg({'rating': ['mean', 'count']})
            popular_genre_movies = popular_genre_movies[popular_genre_movies['rating']['count'] >= min_reviews_threshold]
            popular_genre_movies.columns = ['AverageMovieRating', 'NumReviews']
            
            # Recommend top N movies ordered by ratings in descending order
            recommendations = popular_genre_movies.sort_values(by='AverageMovieRating', ascending=False).head(num_recommendations)
            
            return recommendations.reset_index()

        # Call the popularity-based recommendation function
        genre_recommendations = popularity_recommender(genre, min_reviews_threshold, num_recommendations)

        return render_template('recommendations.html', genre_recommendations=genre_recommendations)

if __name__ == '__main__':
    app.run(debug=True)
