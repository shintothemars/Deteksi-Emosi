<<<<<<< HEAD
import streamlit as st
import joblib
import re
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import warnings
warnings.filterwarnings("ignore")

#load the model
svm_model_emotion = joblib.load("model/svm_best_model.pkl")
svm_model_sentimen = joblib.load("model/best_sentiment_model.pkl")

# laod Vectorizer
vectorizer = joblib.load('vectorizer/tfidf_vectorizer.pkl')

#load encoder
label_encoder_emotion = joblib.load('encoder/label_encoder_emotion.pkl')
label_encoder_sentiment = joblib.load('encoder/label_encoder_sentimen.pkl')

# preprocessing function
def preprocess_text(text): 
    # Lowercase the text
    text = text.lower()
    
    # Remove punctuation and special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.split())  # Remove extra spaces
    # Tokenization
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalpha()]  # Keep only alphabetic tokens

    return ' '.join(tokens)
# streamlit UI
st.title("Emotion and Sentimen Analysis App")
st.write("by: Shinta Arum Imaniyah")

# Input text
user_input = st.text_area("Enter your text here:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.write("Please enter some text for prediction.")
    else:
        # Preprocess the input text
        processed_text = preprocess_text(user_input)
        vectorized_sentimen= vectorizer.transform([processed_text])
        vectorizer_emotion= vectorizer.transform([processed_text])

        sentimen_prediction = svm_model_sentimen.predict(vectorized_sentimen)
        emotion_prediction = svm_model_emotion.predict(vectorizer_emotion)

        sentimen_label = label_encoder_sentiment.inverse_transform(sentimen_prediction)[0]
        emotion_label = label_encoder_emotion.inverse_transform(emotion_prediction)[0]

        #Output the results
        st.subheader("Prediction Results:")
        st.write(f"**Sentimen:** {sentimen_label}")
        st.write(f"**Emotion:** {emotion_label}")

=======
import streamlit as st
import joblib
import re
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import warnings
warnings.filterwarnings("ignore")

#load the model
svm_model_emotion = joblib.load("model/svm_best_model.pkl")
svm_model_sentimen = joblib.load("model/best_sentiment_model.pkl")

# laod Vectorizer
vectorizer = joblib.load('vectorizer/tfidf_vectorizer.pkl')

#load encoder
label_encoder_emotion = joblib.load('encoder/label_encoder_emotion.pkl')
label_encoder_sentiment = joblib.load('encoder/label_encoder_sentimen.pkl')

# preprocessing function
def preprocess_text(text): 
    # Lowercase the text
    text = text.lower()
    
    # Remove punctuation and special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.split())  # Remove extra spaces
    # Tokenization
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalpha()]  # Keep only alphabetic tokens

    return ' '.join(tokens)
# streamlit UI
st.title("Emotion and Sentimen Analysis App")
st.write("by: Shinta Arum Imaniyah")

# Input text
user_input = st.text_area("Enter your text here:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.write("Please enter some text for prediction.")
    else:
        # Preprocess the input text
        processed_text = preprocess_text(user_input)
        vectorized_sentimen= vectorizer.transform([processed_text])
        vectorizer_emotion= vectorizer.transform([processed_text])

        sentimen_prediction = svm_model_sentimen.predict(vectorized_sentimen)
        emotion_prediction = svm_model_emotion.predict(vectorizer_emotion)

        sentimen_label = label_encoder_sentiment.inverse_transform(sentimen_prediction)[0]
        emotion_label = label_encoder_emotion.inverse_transform(emotion_prediction)[0]

        #Output the results
        st.subheader("Prediction Results:")
        st.write(f"**Sentimen:** {sentimen_label}")
        st.write(f"**Emotion:** {emotion_label}")

>>>>>>> b6c48bc8914510a8f67b8bf10d3205c5ff9d5283
        