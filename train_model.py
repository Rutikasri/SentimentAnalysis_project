import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load the existing dataset (Women's Clothing E-Commerce Reviews)
data = pd.read_csv('./Womens Clothing E-Commerce Reviews.csv', encoding='utf-8')

# Load the new dataset (IMDb Movie Reviews)
imdb_data = pd.read_csv('./IMDB Dataset.csv', encoding='utf-8')

# Clean and preprocess both datasets
data = data[['Review Text', 'Rating']].dropna()
imdb_data = imdb_data[['review', 'sentiment']].rename(columns={'review': 'Review Text', 'sentiment': 'Sentiment'}).dropna()

# Convert ratings into sentiment labels for the clothing dataset
def label_sentiment(rating):
    if rating <= 2:
        return 'Negative'
    elif rating == 3:
        return 'Neutral'
    else:
        return 'Positive'

data['Sentiment'] = data['Rating'].apply(label_sentiment)

# Standardize sentiment labels in IMDb dataset
imdb_data['Sentiment'] = imdb_data['Sentiment'].apply(lambda x: x.capitalize())

# Combine the datasets
combined_data = pd.concat([data[['Review Text', 'Sentiment']], imdb_data[['Review Text', 'Sentiment']]], ignore_index=True)

# Balance the training data
class_counts = combined_data['Sentiment'].value_counts()
min_count = class_counts.min()
balanced_data = pd.concat([ 
    combined_data[combined_data['Sentiment'] == 'Positive'].sample(min_count, random_state=42),
    combined_data[combined_data['Sentiment'] == 'Negative'].sample(min_count, random_state=42),
    combined_data[combined_data['Sentiment'] == 'Neutral'].sample(min_count, random_state=42)
])

# Features and labels
X = balanced_data['Review Text']
y = balanced_data['Sentiment']

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Vectorize the text data using TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1, 2))

# Create a pipeline combining TF-IDF and Logistic Regression
model = make_pipeline(vectorizer, LogisticRegression(max_iter=500))

# Train the model
model.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save the model and vectorizer
joblib.dump(model, 'sentiment_model.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

print("Model and LabelEncoder saved successfully!")
