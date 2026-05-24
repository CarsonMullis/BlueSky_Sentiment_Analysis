import pandas as pd
import re
import BSApi

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from collections import Counter
from atproto import models

# Load and preprocess data
df = pd.read_excel(r"C:\Users\colem\Downloads\tweet_sentiment.csv.xlsx")

# Clean tweets
def clean_text(text):
    text = re.sub(r'@\w+|#|http\S+|www\S+|https\S+', '', text.lower())
    return re.sub(r'[^\w\s]', '', text)

tweets = df['tweet'].astype(str).apply(clean_text)
labels = df['label'].astype(str)

# Encode labels
le = LabelEncoder()
encoded_labels = le.fit_transform(labels)

# Split data
tweet_train = tweets.iloc[:10000]
label_train = encoded_labels[:10000]
tweet_test = tweets.iloc[10001:]
label_test = encoded_labels[10001:]

# SBERT embedding
sbert = SentenceTransformer('all-MiniLM-L6-v2')
sX_train = sbert.encode(tweet_train.tolist(), show_progress_bar=True)
sX_test = sbert.encode(tweet_test.tolist(), show_progress_bar=True)

# Scale embeddings
scaler = StandardScaler()
sX_train_scaled = scaler.fit_transform(sX_train)
sX_test_scaled = scaler.transform(sX_test)

# Define models
slr_model = LogisticRegression(max_iter=400)
ssvc_model = SVC(probability=True)
srfc_model = RandomForestClassifier()

# Create soft voting ensemble
ensemble = VotingClassifier(estimators=[
    ('lr', slr_model),
    ('svm', ssvc_model),
    ('rf', srfc_model)
], voting='soft')

# Train ensemble
ensemble.fit(sX_train_scaled, label_train)
ensemble_preds = ensemble.predict(sX_test_scaled)

# Evaluate
print("Ensemble Accuracy:", accuracy_score(label_test, ensemble_preds))

# Confusion matrix
cm = confusion_matrix(label_test, ensemble_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)

plt.figure(figsize=(6, 5))
disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title("Confusion Matrix – SBERT + Ensemble")
plt.tight_layout()
plt.show()

# ==========================
# Predicting on Bluesky Data
# ==========================
print("\n🔵 Fetching and Classifying Live Bluesky Posts...\n")

# Get posts from Bluesky API
real_texts = BSApi.main()

# Clean and embed
clean_real_texts = [clean_text(str(t)) for t in real_texts.values() if isinstance(t, (str, int))]
X_real_sbert = sbert.encode(clean_real_texts, show_progress_bar=False)
X_real_scaled = scaler.transform(X_real_sbert)

# Predict on live data
real_preds = ensemble.predict(X_real_scaled)
probs = ensemble.predict_proba(X_real_scaled)
decoded_preds = le.inverse_transform(real_preds)

# Display predictions
sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
counter = Counter()

print("=== Live Data Predictions (SBERT + Ensemble Voting) ===\n")
for i, (text, pred, conf) in enumerate(zip(real_texts.values(), decoded_preds, probs), 1):
    sentiment_label = sentiment_map[pred]
    counter[sentiment_label] += 1
    print(f"Post {i}:")
    print(f"Text: {text}")
    print(f"Predicted Sentiment: {sentiment_label}")
    print(f"Confidence: {conf}")
    print("-" * 60)

# Summary
print("\n🟢 Bluesky Sentiment Summary:")
for sentiment, count in counter.items():
    print(f"{sentiment}: {count} post(s)")
