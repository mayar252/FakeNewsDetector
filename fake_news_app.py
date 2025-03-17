import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
import requests
from bs4 import BeautifulSoup
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Set Streamlit Page Configuration
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")

# Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("word2vec_cnn_model.h5", compile=False)

model = load_model()

# Load the tokenizer
@st.cache_data
def load_tokenizer():
    with open("tokenizer.pkl", "rb") as handle:
        return pickle.load(handle)

tokenizer = load_tokenizer()

# Define max length for padding (must match training)
MAX_LEN = 515  

# **Streamlit UI Header**
st.markdown("""
    <h1 style='text-align: center; color: #FF4B4B;'>📰 Fake News Detector</h1>
    <h4 style='text-align: center;'>Detect whether a news article is <span style='color:green;'>Real</span> or <span style='color:red;'>Fake</span> using AI</h4>
    <br>
""", unsafe_allow_html=True)

# **Layout: Two Columns for Better UI**
col1, col2 = st.columns([3, 2])

# **Left Column: Input Options**
with col1:
    input_type = st.radio("**Select Input Type:**", ["📄 Type News", "🌍 Enter URL"], horizontal=True)

    if input_type == "📄 Type News":
        news_title = st.text_input("🔹 **Enter News Title:**", placeholder="Enter the news title...")
        news_text = st.text_area("📰 **Enter Full News Content:**", height=200, placeholder="Enter the full news article content here...")

    elif input_type == "🌍 Enter URL":
        url = st.text_input("🌍 **Enter a News Article URL:**", placeholder="Paste the news URL here...")
        news_title = ""
        news_text = ""

# **Single Predict Button**
if st.button("🔍 Predict", use_container_width=True):
    if input_type == "📄 Type News" and (not news_title or not news_text):
        st.warning("⚠️ Please enter both a title and content!")
    elif input_type == "🌍 Enter URL" and not url.strip():
        st.warning("⚠️ Please enter a valid URL!")
    else:
        with st.spinner("🔄 Processing..."):
            # **If URL is Selected, Fetch & Extract Content**
            if input_type == "🌍 Enter URL":
                try:
                    response = requests.get(url)
                    soup = BeautifulSoup(response.text, "html.parser")
                    paragraphs = soup.find_all("p")
                    extracted_text = " ".join([para.get_text() for para in paragraphs])

                    # **GPT-Based Extraction**
                    gpt_prompt = f"Extract the main news article content from the following webpage text:\n\n{extracted_text}"
                    # Simulated GPT Extraction (You Need to Replace This with an API Call)
                    extracted_news = extracted_text[:1500]  # Keeping a reasonable length
                    news_text = extracted_news
                    news_title = "Extracted from URL"
                    st.success("✅ Extracted news content from URL!")
                except:
                    st.error("❌ Failed to fetch news from the provided URL.")
                    news_text = ""

            # **Ensure There is Text to Predict**
            if not news_text:
                st.warning("⚠️ No valid news content available for prediction!")
            else:
                st.markdown("⚙️ **Preprocessing text...**")

                # **Fixing Tokenization & Input Formatting for Accurate Predictions**
                full_text = f"{news_title} {news_text}".strip()
                seq = tokenizer.texts_to_sequences([full_text])
                padded_seq = pad_sequences(seq, maxlen=MAX_LEN, padding="post")

                # Ensure CNN model receives 3D input
                if len(padded_seq.shape) == 2:
                    padded_seq = np.expand_dims(padded_seq, axis=-1)  

                # **Predict with CNN model**
                prediction = model.predict(padded_seq)[0][0]

                # **Display Prediction Results**
                st.markdown("<h3>📊 Prediction Result:</h3>", unsafe_allow_html=True)
                if prediction > 0.5:
                    st.success(f"✅ This news is **REAL** ({prediction*100:.2f}% confidence)")
                    st.progress(float(prediction))
                else:
                    st.error(f"❌ This news is **FAKE** ({(1-prediction)*100:.2f}% confidence)")
                    st.progress(float(1 - prediction))

# **Footer**
st.markdown("---")
st.info("Built with ❤️ using Streamlit & TensorFlow")
