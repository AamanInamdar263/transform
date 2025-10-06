# Step 1: Install transformers (if not already installed)
# pip install transformers torch

from transformers import pipeline

# Step 2: Load a pre-trained sentiment analysis model
sentiment_model = pipeline("sentiment-analysis")

# Step 3: Give the model some text
texts = [
    "I love using Hugging Face Transformers!",
    "This movie was terrible and boring.",
    "The weather is nice today."
]

# Step 4: Predict sentiment for each sentence
for text in texts:
    result = sentiment_model(text)[0]
    print(f"Text: {text}")
    print(f"Label: {result['label']}, Confidence: {result['score']:.4f}\n")
