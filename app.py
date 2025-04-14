import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Download necessary NLTK resources if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Initialize stemmer
ps = PorterStemmer()


# Function to transform text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


# Load the model and vectorizer
@st.cache_resource
def load_models():
    try:
        tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
        model = pickle.load(open('model.pkl', 'rb'))
        return tfidf, model
    except FileNotFoundError:
        st.error("Model files not found. Please ensure 'vectorizer.pkl' and 'model.pkl' are in the current directory.")
        return None, None


# Set page configuration
st.set_page_config(
    page_title="Spam Guard - Email/SMS Spam Classifier",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 42px;
        font-weight: bold;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 20px;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 24px;
        font-weight: 500;
        color: #475569;
        margin-bottom: 20px;
    }
    .result-spam {
        padding: 20px;
        border-radius: 10px;
        font-weight: bold;
        font-size: 28px;
        color: white;
        background-color: #EF4444;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .result-ham {
        padding: 20px;
        border-radius: 10px;
        font-weight: bold;
        font-size: 28px;
        color: white;
        background-color: #10B981;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stTextInput > div > div > input {
        border-radius: 8px;
    }
    .stButton > button {
        background-color: #2563EB;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
        border: none;
        width: 100%;
    }
    .stButton > button:hover {
        background-color: #1D4ED8;
    }
    .info-box {
        background-color: #F3F4F6;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
    }
    .text-area-header {
        font-size: 18px;
        font-weight: 500;
        color: #1F2937;
    }
    .highlighted-text {
        background-color: #FFFBEB;
        padding: 5px;
        border-radius: 4px;
        font-style: italic;
    }
    .footer {
        text-align: center;
        margin-top: 60px;
        color: #6B7280;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar content
with st.sidebar:
    st.markdown("<h1 style='text-align: center;'>About Spam Guard</h1>", unsafe_allow_html=True)
    st.markdown("""
    **Spam Guard** is an intelligent system that uses machine learning to 
    classify messages as spam or legitimate communications.

    ### How it works:
    1. **Text Preprocessing**: Converts your message into a standardized format
    2. **Feature Extraction**: Identifies key patterns in the text
    3. **ML Classification**: Evaluates the likelihood of spam
    4. **Result Display**: Shows you the classification with confidence score

    ### Common spam indicators:
    - Urgency or pressure tactics
    - Requests for personal information
    - Unusual links or attachments
    - Grammatical errors or strange formatting
    - Offers that seem too good to be true
    """)

    st.markdown("---")
    st.markdown("### Sample Messages to Test")

    if st.button("📱 Try Example 1 (Spam)"):
        example_spam = "URGENT: You have WON a £1,000 cash prize! Call 09061701461 from landline. Claim S89. Valid 12hrs only."
        st.session_state['example_text'] = example_spam

    if st.button("📱 Try Example 2 (Not Spam)"):
        example_ham = "Hey, can we meet at the coffee shop around 5pm today? I need to discuss the project details."
        st.session_state['example_text'] = example_ham

# Main content
col1, col2, col3 = st.columns([1, 8, 1])

with col2:
    st.markdown("<h1 class='main-header'>🛡️ Spam Guard</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Protect yourself from unwanted messages with AI-powered detection</p>",
                unsafe_allow_html=True)

    # Text input area
    st.markdown("<p class='text-area-header'>Enter the message to analyze:</p>", unsafe_allow_html=True)

    # Use session state to maintain example text
    if 'example_text' not in st.session_state:
        st.session_state['example_text'] = ""

    input_sms = st.text_area("", value=st.session_state['example_text'], height=150, key="message_input")

    analyze_col, clear_col = st.columns([3, 1])

    with analyze_col:
        predict_button = st.button("🔍 Analyze Message", use_container_width=True)

    with clear_col:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state['example_text'] = ""
            st.rerun()


    # Load models
    tfidf, model = load_models()

    # Make prediction when button is clicked
    if predict_button and input_sms and tfidf is not None and model is not None:
        with st.spinner('Analyzing the message...'):
            # Create a progress bar for visual effect
            progress_bar = st.progress(0)
            for i in range(100):
                # Update progress bar
                progress_bar.progress(i + 1)
                time.sleep(0.01)

            # 1. Preprocess
            transformed_sms = transform_text(input_sms)

            # 2. Vectorize
            vector_input = tfidf.transform([transformed_sms])

            # 3. Predict
            result = model.predict(vector_input)[0]

            # Get prediction probability
            prediction_proba = model.predict_proba(vector_input)[0]
            confidence = max(prediction_proba) * 100

            # 4. Display
            if result == 1:
                st.markdown(
                    f"<div class='result-spam'>⚠️ SPAM DETECTED<br><span style='font-size: 18px;'>Confidence: {confidence:.2f}%</span></div>",
                    unsafe_allow_html=True)

                # Display warning information for spam
                st.markdown("""
                <div class='info-box'>
                    <h3>⚠️ Warning: This message appears to be spam!</h3>
                    <p>We recommend:</p>
                    <ul>
                        <li>Do not respond to the message</li>
                        <li>Do not click any links contained in the message</li>
                        <li>Do not provide personal information</li>
                        <li>Report the message to your provider if possible</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(
                    f"<div class='result-ham'>✅ NOT SPAM<br><span style='font-size: 18px;'>Confidence: {confidence:.2f}%</span></div>",
                    unsafe_allow_html=True)

                # Display information for legitimate messages
                st.markdown("""
                <div class='info-box'>
                    <h3>✅ This message appears to be legitimate</h3>
                    <p>While our analysis suggests this is not spam, always use caution when:</p>
                    <ul>
                        <li>Clicking on links from unfamiliar senders</li>
                        <li>Sharing personal or financial information</li>
                        <li>Receiving unexpected requests, even from known contacts</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

            # Show preprocessing steps for educational purposes
            with st.expander("🔍 See analysis details"):
                st.subheader("Text Preprocessing Steps:")
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Original Text:**")
                    st.markdown(f"<div class='highlighted-text'>{input_sms}</div>", unsafe_allow_html=True)

                with col2:
                    st.markdown("**Processed Text:**")
                    st.markdown(f"<div class='highlighted-text'>{transformed_sms}</div>", unsafe_allow_html=True)

                # Create wordcloud-like visual representation of key terms
                words = transformed_sms.split()
                if words:
                    word_freq = {}
                    for word in words:
                        if word in word_freq:
                            word_freq[word] += 1
                        else:
                            word_freq[word] = 1

                    # Create dataframe for visualization
                    word_df = pd.DataFrame(list(word_freq.items()), columns=['Word', 'Frequency'])
                    if not word_df.empty:
                        st.subheader("Key Terms Detected:")
                        fig, ax = plt.subplots(figsize=(10, 4))
                        ax.barh(word_df['Word'], word_df['Frequency'], color='#3B82F6')
                        ax.set_xlabel('Frequency')
                        ax.set_title('Word Frequency in Message')
                        st.pyplot(fig)

    # Display footer
    st.markdown("<div class='footer'>© 2025 Spam Guard | Email/SMS Spam Classification System</div>",
                unsafe_allow_html=True)