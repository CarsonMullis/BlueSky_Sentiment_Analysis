BlueskyProj — Real-Time Sentiment Analysis on Bluesky Posts
A machine learning pipeline that fetches live posts from the Bluesky social network and classifies their sentiment in real time using a SBERT embedding model combined with a soft-voting ensemble classifier.

Overview
This project was built to explore sentiment analysis on real-time social media data outside of traditional platforms like Twitter/X. It trains a multi-model ensemble on a labeled tweet dataset, then applies the trained classifier to live Bluesky posts fetched via the Bluesky API — outputting per-post sentiment predictions with confidence scores and an overall summary.

Features

Text Preprocessing — Strips mentions, hashtags, URLs, and punctuation from raw post text
SBERT Embeddings — Encodes tweets and live posts using all-MiniLM-L6-v2 for semantically rich feature vectors
Soft Voting Ensemble — Combines Logistic Regression, SVM, and Random Forest classifiers for improved accuracy
Live Bluesky Integration — Fetches real posts from a Bluesky home feed via a custom API module
Confidence Scoring — Outputs per-post prediction probabilities alongside sentiment labels
Confusion Matrix Visualization — Evaluates model performance on the test set with a labeled confusion matrix
