import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Load the pre-trained model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Streamlit Interface
st.title("Sentiment Analysis with BERT")

# User input for prediction
text = st.text_area("Enter text for sentiment analysis:")

if text:
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=-1).item()

    # Display the result
    sentiment = "Positive" if predicted_class == 1 else "Negative"
    st.write(f"Predicted Sentiment: {sentiment}")
