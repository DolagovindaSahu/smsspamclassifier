import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import numpy as np

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for word in text:
        if word not in stopwords.words('english') and word not in string.punctuation:
            y.append(word)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


tfidf = pickle.load(open('Vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title('Email/SMS Spam Classifier')

input_sms = st.text_area('Enter The Message')

if st.button('Predict'):
    # 1. Preprocess
    transformed_sms = transform_text(input_sms)
    # 2. Vectorize
    vector_input = tfidf.transform([transformed_sms])
    # print(type(vector_input))
    # number of character column adding to the vector_input
    length_of_sms = len(input_sms)
    combined_vector = np.append(vector_input.toarray(),length_of_sms).reshape(1, -1)
    # 3. Predict
    result = model.predict(combined_vector)[0]
    # 4. display
    if result == 1:
        st.header('Spam')
    else:
        st.header('Not Spam')
