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
    st.set_page_config(
        page_title="Chatbot",  # Set the page title
        page_icon="🤖",         # Add an emoji to the browser tab
        layout="wide",          # Use wide layout for the app
        initial_sidebar_state="expanded"  # Keep the sidebar open by default
    )

    # Add custom CSS styling for the chatbot interface
    st.markdown("""
        <style>
            .chatbot-container {
                background-color: #f9f9f9;
                padding: 20px;
                border-radius: 15px;
                box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
            }
            .user-message {
                background-color: #c1e1f5;
                padding: 10px;
                border-radius: 10px;
                margin: 5px;
                text-align: left;
                max-width: 70%;
                margin-left: 20px;
            }
            .chatbot-message {
                background-color: #e0e0e0;
                padding: 10px;
                border-radius: 10px;
                margin: 5px;
                text-align: right;
                max-width: 70%;
                margin-right: 20px;
            }
            .sidebar .sidebar-content {
                background-color: #e0f7fa;
            }
            .stTextInput input {
                background-color: #e3f2fd;
            }
        </style>
    """, unsafe_allow_html=True)

    # Sidebar for navigation
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
            # Displaying user message and chatbot response in a chat-like style
            st.markdown(f'<div class="user-message">{user_input}</div>', unsafe_allow_html=True)

            response = chatbot(user_input)
            st.markdown(f'<div class="chatbot-message">{response}</div>', unsafe_allow_html=True)

            # Log the conversation
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
