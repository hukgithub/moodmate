import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from gtts import gTTS
import os
import random
from datetime import datetime

# Sentiment analysis model
sentiment_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Text generation model
model_name = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
text_gen_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Function to analyze mood based on user input
def analyze_mood(text):
    result = sentiment_model(text)
    mood = result[0]["label"]
    return mood

# Function to generate personalized content based on mood
# Update the generate_content function
# Function to generate predefined responses based on mood
def generate_content(mood):
    # Dictionary of predefined messages for each mood
    responses = {
        "POSITIVE": [
            "That's wonderful! Remember to carry that joy with you throughout the day.",
            "You seem happy! Keep spreading those good vibes around!",
            "Feeling great? That's amazing! Stay positive and keep shining!"
        ],
        "NEGATIVE": [
            "I'm here for you. Take a deep breath, and remember that tough times pass.",
            "It's okay to feel down sometimes. You're stronger than you think.",
            "Remember, it's alright to feel sad. Take your time, and things will get better."
        ],
        "NEUTRAL": [
            "Every day brings new opportunities. Stay mindful and take it one step at a time.",
            "Balance is key. Hope you have a peaceful and fulfilling day ahead.",
            "Remember to take moments today to check in with yourself. You're doing great."
        ]
    }
    
    # Select a random message from the appropriate mood list
    return random.choice(responses.get(mood, responses["NEUTRAL"]))


# Function for Text-to-Speech (TTS)
def text_to_speech(text, filename="output.mp3"):
    tts = gTTS(text=text, lang='en')
    tts.save(filename)
    return filename

# Streamlit app layout
st.title("MoodMate: Personalized Wellness Support")
st.write("Your daily wellness companion for personalized emotional support.")

# Daily check-in
st.subheader("How are you feeling today?")
user_input = st.text_input("Enter your thoughts or feelings here:")

# Analyze and respond based on mood
if user_input:
    mood = analyze_mood(user_input)
    st.write(f"Detected mood: {mood}")

    # Generate content
    content = generate_content(mood)
    st.write("Here's something for you:")
    st.write(content)

    # Option to listen to the content
    if st.button("Listen to this message"):
        audio_file = text_to_speech(content)
        st.audio(audio_file, format="audio/mp3")
