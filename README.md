# CineMatch-Movie-Recommendation-System
A content-based movie recommendation engine built with Python and Scikit-Learn. Utilizes NLP techniques and cosine similarity to suggest personalized film recommendations based on genres, cast, crew, and plot keywords.


🎬 CineMatch: Movie Recommendation System

A content-based movie recommender built with Python, scikit-learn and Streamlit. Pick a movie you like and get similar titles, ranked by a similarity score, in an animated, dark-themed interface.

<!-- Add a screenshot: save it as assets/screenshot.png and uncomment the next line --> <!-- ![CineMatch screenshot](assets/screenshot.png) -->

🔗 Live demo: add your Streamlit link here after deploying

✨ Features
Search any of ~4,800 movies from the TMDB 5000 dataset
Top-N recommendations (3–15) with a % similarity bar
Rating, release year, genres and overview on every card
Animated gradient UI, hover effects and responsive card grid
Fast startup: only a compact neighbors.npz (< 1 MB) is loaded
🧠 How it works
Merge the TMDB movies and credits datasets.
Build a tags field for each movie: overview + genres + keywords + director.
Clean and stem the text (Porter stemmer), then vectorise it with CountVectorizer (5,000 features, English stop words removed).
Compute cosine similarity between all movie vectors.
For a chosen movie, return the closest matches.

The full 4806 × 4806 similarity matrix is about 185 MB, which is too big for GitHub. The app only displays the top 15 matches, so it keeps each movie's 30 nearest neighbours (neighbors.npz). The recommendations are identical.

🗂️ Project structure
├── app.py                  # Streamlit app
├── movie_list.pkl          # movie ids, titles and tags
├── neighbors.npz           # top-30 similar movies per movie (built from similarity.pkl)
├── tmdb_5000_movies.csv    # optional: ratings, genres, overviews for the cards
├── requirements.txt
├── .streamlit/config.toml  # dark theme
└── notebook/               # model-building notebook (optional)


Python · Pandas · NumPy · scikit-learn · NLTK · Streamlit

📊 Data

TMDB 5000 Movie Dataset on Kaggle. This product uses TMDB data but is not endorsed or certified by TMDB.

📄 License

MIT. Add a LICENSE file if you want others to reuse the code.
