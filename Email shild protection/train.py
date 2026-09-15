import pandas as pd
import numpy as np
import nltk
import string
import pickle
import os
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    # Keep alphanumeric characters
    y = [i for i in text if i.isalnum()]
    
    # Remove stopwords and punctuation
    text = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]
    
    # Apply Stemming
    y = [ps.stem(i) for i in text]
    
    return " ".join(y)

# Locate dataset in the same directory as this script
script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(script_dir, 'spam.csv')

# 1. Load Dataset
df = pd.read_csv(dataset_path, encoding='latin-1')

# Keep relevant columns and rename
df = df[['v1', 'v2']]
df.columns = ['target', 'text']

# Map target explicitly (ham -> 0, spam -> 1)
df['target'] = df['target'].map({'ham': 0, 'spam': 1})

# 2. Preprocess text
df['transformed_text'] = df['text'].apply(transform_text)

# 3. Vectorize text
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df['transformed_text']).toarray()
y = df['target'].values

# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# 5. Train Model
mnb = MultinomialNB()
mnb.fit(X_train, y_train)

# Print performance metrics
y_pred = mnb.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")

# 6. Save model and vectorizer files
pickle.dump(tfidf, open(os.path.join(script_dir, 'vectorizer.pkl'), 'wb'))
pickle.dump(mnb, open(os.path.join(script_dir, 'model.pkl'), 'wb'))
print("Successfully saved vectorizer.pkl and model.pkl!")
