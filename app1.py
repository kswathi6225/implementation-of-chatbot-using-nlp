
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
    st.title("Chatbot Using NLP and Logistic Regression")

    menu = ["Home", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.write("Start chatting with the bot!")
        
        if not os.path.exists("chat_log.csv"):
            with open("chat_log.csv", "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["User Input", "Chatbot Response", "Timestamp"])
        
        user_input = st.text_input("You: ")
        if user_input:
            response = chatbot(user_input)
            st.write(f"Chatbot: {response}")
            
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
        st.write("This chatbot is powered by NLP and Logistic Regression.")

if __name__ == "__main__":
    main()
