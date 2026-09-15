import streamlit as st
import pickle
import string
import nltk
import os
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = [i for i in text if i.isalnum()]
    text = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]
    y = [ps.stem(i) for i in text]
    
    return " ".join(y)

script_dir = os.path.dirname(os.path.abspath(__file__))

# Load model and vectorizer
tfidf = pickle.load(open(os.path.join(script_dir, 'vectorizer.pkl'), 'rb'))
model = pickle.load(open(os.path.join(script_dir, 'model.pkl'), 'rb'))

st.title("SMS / Email Spam Classifier")

input_sms = st.text_area("Enter the message below:")

if st.button('Predict'):
    if not input_sms.strip():
        st.warning("Please enter a valid message.")
    else:
        # Preprocess and predict
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]
        
        # Display results (0 = Ham, 1 = Spam)
        if result == 1:
            st.error("🚨 This message is **SPAM**!")
        else:
            st.success("✅ This message is **NOT SPAM (Ham)**.")