import streamlit as st
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
import webbrowser
from keras.models import load_model
model = load_model('chatbot_model.h5')
import json
import random

intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))


def clean_up_sentence(sentence):
    # tokenize the pattern - splitting words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stemming every word - reducing to base form
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


# return bag of words array: 0 or 1 for words that exist in sentence
def bag_of_words(sentence, words, show_details=True):
    # tokenizing patterns
    sentence_words = clean_up_sentence(sentence)
    # bag of words - vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,word in enumerate(words):
            if word == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % word)
    return(np.array(bag))



st.title('Hey I am G your friendly chatbot')

# Function to predict the class and generate a response
def predict_class(msg):
    # Load the JSON file with intents
    with open('intents.json', 'r') as file:
        intents_data = json.load(file)

    # Implement your prediction logic here
    # For simplicity, we'll assume a predefined intent based on user input.
    for intent in intents_data['intents']:
        if msg.lower() in intent['patterns']:
            return intent['tag']
    return "unknown"

# Function to get a response based on predicted intent
def getResponse(intent):
    # Load the JSON file with intents
    with open('intents.json', 'r') as file:
        intents_data = json.load(file)

    for intent_info in intents_data['intents']:
        if intent_info['tag'] == intent:
            responses = intent_info['responses']
            return random.choice(responses)
    return "I'm not sure how to respond to that."

# Create a text input box
user_input = st.text_area("You:", "").strip()

# Create a "Send" button
if st.button("Send"):
    predicted_intent = predict_class(user_input)
    res = getResponse(predicted_intent)
    if(res[0:5]=='https'):
        webbrowser.open(res)
    st.text("G: " + res)

