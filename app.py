import streamlit as st
import pickle
import re
import ssl
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from newspaper import Article
from urllib.parse import urlparse
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Download NLTK data (you might need to handle SSL issues here)
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download("stopwords")
nltk.download("wordnet")

# Constants
MAX_SEQUENCE_LENGTH = 200

# Load the models and vectorizer
@st.cache_resource
def load_model(model_name):
    model_path = f"models/Liar/{model_name}_Liar_model.pkl"
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error(f"Model file not found: {model_path}")
        return None

@st.cache_resource
def load_vectorizer():
    vectorizer_path = "models/Liar/Tfidf_Liar_vectorizer.pkl"
    try:
        with open(vectorizer_path, "rb") as f:
            vectorizer = pickle.load(f)
        return vectorizer
    except FileNotFoundError:
        st.error(f"Vectorizer file not found: {vectorizer_path}")
        return None

@st.cache_resource
def load_tokenizer(model_type):
    if model_type == "LSTM":
        tokenizer_path = "models/Liar/LSTM_Liar_tokenizer.pkl"
    else:  # BiLSTM
        tokenizer_path = "models/Liar/BiLSTM_Liar_tokenizer.pkl"
    
    try:
        with open(tokenizer_path, "rb") as f:
            tokenizer = pickle.load(f)
        return tokenizer
    except FileNotFoundError:
        st.error(f"Tokenizer file not found: {tokenizer_path}")
        return None

def extract_article_text(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        st.error(f"Error extracting article: {str(e)}")
        return None

# Preprocess text function (same as in your notebook)
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    words = [
        lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words
    ]
    return " ".join(words)

def analyze_text(text, model, model_type, vectorizer=None, tokenizer=None):
    # Preprocess the text
    processed_text = preprocess_text(text)

    if model_type in ["LSTM", "BiLSTM"]:
        if tokenizer is None:
            st.error("Tokenizer is required for LSTM/BiLSTM models")
            return None, None
        # Process and predict using LSTM/BiLSTM
        sequence = tokenizer.texts_to_sequences([processed_text])
        padded_sequence = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
        raw_prediction = float(model.predict(padded_sequence)[0][0])
        prediction = 1 if raw_prediction > 0.5 else 0
        probability = raw_prediction if prediction == 1 else 1 - raw_prediction
    else:
        if vectorizer is None:
            st.error("Vectorizer is required for traditional ML models")
            return None, None
        # Vectorize and predict using other models
        text_vector = vectorizer.transform([processed_text])
        prediction = model.predict(text_vector)[0]
        probability = float(model.predict_proba(text_vector)[0][prediction])

    return prediction, probability

# Main app
def main():
    st.title("📰 NewsCheck - Fake News Detector")
    st.markdown(
        """
    This app uses machine learning to detect whether a news article is likely to be real or fake.
    Choose your input method below and enter the text or URL to check its authenticity.
    """
    )

    # Model selection
    model_type = st.selectbox(
        "Select Model",
        ["LogisticRegression", "RandomForest", "GradientBoosting", "NaiveBayes", "LSTM", "BiLSTM"],
        help="Choose between different models for fake news detection"
    )

    # Load appropriate model and preprocessing components
    model = load_model(model_type)
    if model is None:
        st.error("Failed to load model. Please try a different model.")
        return

    if model_type in ["LSTM", "BiLSTM"]:
        tokenizer = load_tokenizer(model_type)
        if tokenizer is None:
            st.error("Failed to load tokenizer. Please try a different model.")
            return
        vectorizer = None
    else:
        vectorizer = load_vectorizer()
        if vectorizer is None:
            st.error("Failed to load vectorizer. Please try a different model.")
            return
        tokenizer = None

    # Input method selection
    input_method = st.radio(
        "Choose input method:",
        ["Enter URL", "Enter Text"],
        horizontal=True
    )

    if input_method == "Enter URL":
        # URL input
        url_input = st.text_input("Enter news article URL:")
        
        if st.button("Check News"):
            if url_input:
                # Validate URL
                try:
                    result = urlparse(url_input)
                    if not all([result.scheme, result.netloc]):
                        st.error("Please enter a valid URL")
                        return
                except:
                    st.error("Please enter a valid URL")
                    return

                with st.spinner("Extracting and analyzing article..."):
                    # Extract article text
                    article_text = extract_article_text(url_input)

                    if article_text:
                        # Analyze the text
                        prediction, probability = analyze_text(
                            article_text, 
                            model, 
                            model_type, 
                            vectorizer,
                            tokenizer
                        )

                        if prediction is None or probability is None:
                            return

                        # Display results
                        st.markdown("---")
                        st.subheader("Results")

                        if prediction == 1:
                            st.success(
                                f"✅ This appears to be REAL NEWS (Confidence: {probability:.2%})"
                            )
                        else:
                            st.error(
                                f"❌ This appears to be FAKE NEWS (Confidence: {probability:.2%})"
                            )

                        # Show confidence meter
                        st.progress(float(probability))
                        st.caption(f"Confidence: {probability:.2%}")

                        # Display extracted text
                        with st.expander("View extracted article text"):
                            st.text(article_text)
            else:
                st.warning("Please enter a URL to analyze.")
    else:
        # Direct text input
        text_input = st.text_area("Enter news article text:", height=200)
        
        if st.button("Analyze Text"):
            if text_input:
                with st.spinner("Analyzing text..."):
                    # Analyze the text
                    prediction, probability = analyze_text(
                        text_input, 
                        model, 
                        model_type, 
                        vectorizer,
                        tokenizer
                    )

                    if prediction is None or probability is None:
                        return

                    # Display results
                    st.markdown("---")
                    st.subheader("Results")

                    if prediction == 1:
                        st.success(
                            f"✅ This appears to be REAL NEWS (Confidence: {probability:.2%})"
                        )
                    else:
                        st.error(
                            f"❌ This appears to be FAKE NEWS (Confidence: {probability:.2%})"
                        )

                    # Show confidence meter
                    st.progress(float(probability))
                    st.caption(f"Confidence: {probability:.2%}")
            else:
                st.warning("Please enter some text to analyze.")

if __name__ == "__main__":
    main()
