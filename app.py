import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download required files (first time only)
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []

    # Keep only alphanumeric words
    for word in text:
        if word.isalnum():
            y.append(word)

    text = y
    y = []

    # Remove stopwords
    for word in text:
        if word not in stop_words and word not in string.punctuation:
            y.append(word)

    text = y
    y = []

    # Stemming
    for word in text:
        y.append(ps.stem(word))

    return " ".join(y)

# Load saved model
tfidf = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

# App Title
st.title("SMS Spam Classifier")

# User Input
input_sms = st.text_area("Enter your message")

if st.button("Predict"):

    transformed_sms = transform_text(input_sms)

    vector_input = tfidf.transform([transformed_sms])

    prediction = model.predict(vector_input)[0]

    if prediction == 1:
        st.error("Spam Message")
    else:
        st.success("Not Spam")