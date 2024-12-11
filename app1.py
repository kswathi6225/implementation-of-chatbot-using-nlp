import os
import json
import datetime
import csv
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Configure SSL for nltk
ssl._create_default_https_context = ssl._create_unverified_context
nltk.download('punkt')

# Load intents from the JSON file
with open("intents.json", "r") as file:
    intents = json.load(file)

# Preprocess data for training
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# Create vectorizer and classifier
vectorizer = TfidfVectorizer(ngram_range=(1, 4))
x = vectorizer.fit_transform(patterns)
y = tags

clf = LogisticRegression(random_state=0, max_iter=10000)
clf.fit(x, y)

# Chatbot function
def chatbot(input_text):
    input_vector = vectorizer.transform([input_text])
    tag = clf.predict(input_vector)[0]
    for intent in intents:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])

# Streamlit interface
def main():
    # Set page configuration
    st.set_page_config(
        page_title="Chatbot",  # Page title
        page_icon="🤖",         # Emoji icon in the tab
        layout="wide",          # Wide layout for spacious appearance
        initial_sidebar_state="expanded"  # Keep the sidebar open by default
    )

    # Custom CSS styling for green and black theme
    st.markdown("""
        <style>
            .user-message {
                background-color: #8BC34A;  /* Green background */
                color: white;
                padding: 10px;
                border-radius: 10px;
                margin: 5px;
                text-align: left;
                max-width: 70%;
                margin-left: 20px;
            }
            .chatbot-message {
                background-color: #212121;  /* Black background */
                color: white;
                padding: 10px;
                border-radius: 10px;
                margin: 5px;
                text-align: right;
                max-width: 70%;
                margin-right: 20px;
            }
            body {
                background-image: url('https://yourimageurl.com/image.jpg');  /* Replace with your image URL */
                background-size: cover;
            }
        </style>
    """, unsafe_allow_html=True)

    # Sidebar menu
    menu = ["Home", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.write("<h2>Start chatting with the bot!</h2>", unsafe_allow_html=True)

        if not os.path.exists("chat_log.csv"):
            with open("chat_log.csv", "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["User Input", "Chatbot Response", "Timestamp"])

        user_input = st.text_input("You: ", key="user_input")
        
        if user_input:
            # Display user message in chat bubble style
            st.markdown(f'<div class="user-message">{user_input}</div>', unsafe_allow_html=True)

            response = chatbot(user_input)
            # Display chatbot response in chat bubble style
            st.markdown(f'<div class="chatbot-message">{response}</div>', unsafe_allow_html=True)

            # Log the conversation to the chat_log.csv
            with open("chat_log.csv", "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([user_input, response, datetime.datetime.now()])

    elif choice == "Conversation History":
        st.write("Conversation History:")
        if os.path.exists("chat_log.csv"):
            with open("chat_log.csv", "r") as file:
                reader = csv.reader(file)
                for row in reader:
                    st.write(f"User: {row[0]} | Chatbot: {row[1]} | Time: {row[2]}")
        else:
            st.write("No history available.")

    elif choice == "About":
        st.write("""
            This chatbot is powered by **Natural Language Processing (NLP)** and **Logistic Regression**.
            It predicts user intent based on predefined patterns and responds accordingly.
        """)

if __name__ == "__main__":
    main()
