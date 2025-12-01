import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

# Page config
st.set_page_config(page_title="NLP Finance Sentiment Analysis", page_icon="📈")

# Title
st.title("NLP Finance Sentiment Analysis")
st.markdown("Financial sentiment analysis app")
st.markdown("Deployed from Colab notebook")

# Cache the model loading so it only happens once
@st.cache_resource
def load_model():
    # Replace with your specific model
    # If you fine-tuned your own model, upload it to Hugging Face and use that path
    model_name = "ProsusAI/finbert"  # Popular financial BERT model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

# Load model
try:
    with st.spinner("Loading model..."):
        tokenizer, model = load_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Function to analyze sentiment
def analyze_sentiment(text):
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get probabilities
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    probs = probs.detach().numpy()[0]
    
    # Get prediction
    prediction = np.argmax(probs)
    
    # Map to labels (adjust based on your model)
    labels = ["Negative", "Neutral", "Positive"]
    
    return labels[prediction], probs

# Text input
st.subheader("Enter Financial Text to Analyze")
text_input = st.text_area(
    "Input text:",
    placeholder="Example: The company reported strong quarterly earnings, exceeding analyst expectations...",
    height=150
)

# Analyze button
if st.button("Analyze Sentiment", type="primary"):
    if text_input.strip():
        with st.spinner("Analyzing..."):
            sentiment, probabilities = analyze_sentiment(text_input)
        
        # Display results
        st.subheader("Results")
        
        # Show sentiment with color
        if sentiment == "Positive":
            st.success(f"**Sentiment: {sentiment}** 📈")
        elif sentiment == "Negative":
            st.error(f"**Sentiment: {sentiment}** 📉")
        else:
            st.info(f"**Sentiment: {sentiment}** ➡️")
        
        # Show confidence scores
        st.subheader("Confidence Scores")
        labels = ["Negative", "Neutral", "Positive"]
        for label, prob in zip(labels, probabilities):
            st.write(f"{label}: {prob:.2%}")
            st.progress(float(prob))
    else:
        st.warning("Please enter some text to analyze!")

# Example texts
with st.expander("Try Example Texts"):
    examples = [
        "The stock price surged after the company announced record-breaking profits.",
        "The company faces potential bankruptcy due to mounting debts.",
        "The quarterly report showed steady performance with no major changes."
    ]
    
    for i, example in enumerate(examples, 1):
        if st.button(f"Example {i}", key=f"ex{i}"):
            st.session_state.example_text = example
            st.rerun()

# Use example if selected
if 'example_text' in st.session_state:
    st.text_area("Input text:", value=st.session_state.example_text, height=150, key="example_display")
    del st.session_state.example_text

# Footer
st.markdown("---")
st.markdown("Built with Streamlit and 🤗 Transformers")
