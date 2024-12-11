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
    # Set the page configuration for Streamlit
    st.set_page_config(
        page_title="Chatbot",  # Page title
        page_icon="🤖",         # Emoji icon
        layout="wide",          # Wide layout for better appearance
        initial_sidebar_state="expanded"  # Sidebar open by default
    )

    # Custom CSS for green and black theme and sidebar styling
    st.markdown("""
        <style>
            /* Chat message styling */
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
            /* Chatbox at the bottom */
            .chatbox-container {
                position: fixed;
                bottom: 20px;
                left: 20px;
                right: 20px;
                padding: 10px;
                background-color: #f0f0f0;
                border-radius: 10px;
                box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.2);
            }
            /* Sidebar customizations */
            .sidebar .sidebar-content {
                background-color: #2C3E50;  /* Darker color for sidebar */
            }
            .sidebar .sidebar-content .stButton button {
                background-color: #16A085;  /* Green button color */
                color: white;
                border-radius: 10px;
                padding: 10px;
                width: 100%;
                margin-bottom: 10px;
            }
            .sidebar .sidebar-content .stButton button:hover {
                background-color: #1abc9c;
            }
            /* New Chat Button in the top-right */
            .new-chat-btn {
                position: absolute;
                top: 10px;
                right: 20px;
                background-color: #FF6347;  /* Tomato color for the button */
                color: white;
                padding: 10px;
                border-radius: 50%;
                font-size: 20px;
                cursor: pointer;
                border: none;
            }
            .new-chat-btn:hover {
                background-color: #ff4500;
            }
            /* Robot Icon inside the chat */
            .robot-icon {
                font-size: 30px;
                color: #8BC34A;
                margin-right: 10px;
                vertical-align: middle;
            }
            /* Adding more items to the left sidebar */
            .sidebar-content-more {
                padding-top: 20px;
                color: white;
                font-size: 18px;
            }
            .sidebar-content-more a {
                color: #16A085;
                text-decoration: none;
            }
            .sidebar-content-more a:hover {
                color: #1abc9c;
            }
        </style>
    """, unsafe_allow_html=True)

    # Sidebar menu
    menu = ["Home", "History", "New Chat", "About"]
    choice = st.sidebar.selectbox("Menu", menu, index=0)

    # Add extra information in the sidebar
    st.sidebar.markdown("<div class='sidebar-content-more'>", unsafe_allow_html=True)
    st.sidebar.markdown("<h4>About the Chatbot</h4>", unsafe_allow_html=True)
    st.sidebar.markdown("<a href='#'>Chatbot Info</a>", unsafe_allow_html=True)
    st.sidebar.markdown("<a href='#'>Features</a>", unsafe_allow_html=True)
    st.sidebar.markdown("</div>", unsafe_allow_html=True)

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
            # Display chatbot response in chat bubble style with robot icon
            st.markdown(f'<div class="chatbot-message"><span class="robot-icon">🤖</span>{response}</div>', unsafe_allow_html=True)

            # Log the conversation
            with open("chat_log.csv", "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([user_input, response, datetime.datetime.now()])

    elif choice == "History":
        st.write("Conversation History:")
        if os.path.exists("chat_log.csv"):
            with open("chat_log.csv", "r") as file:
                reader = csv.reader(file)
                for row in reader:
                    st.write(f"User: {row[0]} | Chatbot: {row[1]} | Time: {row[2]}")
        else:
            st.write("No history available.")

    elif choice == "New Chat":
        # Add the New Chat button in the right-top corner of the page
        st.markdown('<button class="new-chat-btn">+</button>', unsafe_allow_html=True)
        st.write("<h2>Start a new conversation with the bot!</h2>", unsafe_allow_html=True)
        st.text_input("You: ", key="new_chat")

    elif choice == "About":
        st.write("""
            This chatbot is powered by **Natural Language Processing (NLP)** and **Logistic Regression**.
            It predicts user intent based on predefined patterns and responds accordingly.
        """)

if __name__ == "__main__":
    main()
