#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


csv_path='processed11_data.xlsx'
data=pd.read_excel(csv_path)


# In[3]:


data.head(20)


# In[4]:


#Necessary Packages for Classification in Navie-Bayes
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer        #Used For Converting text into Numerical(Binary Data)
from sklearn.model_selection import train_test_split               #Used for splitting dataset for Training and testing the performance
from sklearn.naive_bayes     import MultinomialNB                  #Used for text based Classification & identify the intent
from sklearn.metrics   import accuracy_score,classification_report


# In[5]:


#Create a DataFrame
df=pd.DataFrame(data)
df[['intent','stemmed_text']]


# In[6]:


#Spliting the data into training & Testing Sets
X_train,X_test,y_train,y_test=train_test_split(df['stemmed_text'],df['intent'],test_size=0.2, random_state=42)

# Vectorize the text data using CountVectorizer (you can also use TfidfVectorizer)
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Create and train the Multinomial Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_vectorized, y_train)

# Make predictions on the test data
y_pred = nb_classifier.predict(X_test_vectorized)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
classification_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report:\n", classification_report)


# In[7]:


# Vectorize the text data using CountVectorizer (you can also use TfidfVectorizer)
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)       #Convert X_train into Numerical by fiting & transform
X_test_vectorized = vectorizer.transform(X_test)             #Convert X_test into Numerical


# In[8]:


# Create and train the Multinomial Naive Bayes classifier
nb_classifier = MultinomialNB()
#Training the X_train(numerical) &  y_train(intent) data with MultiNomial
nb_classifier.fit(X_train_vectorized, y_train)                        

# Make predictions on the test data
y_pred = nb_classifier.predict(X_test_vectorized)          


# In[9]:


#Evaluate the Classifier
accuracy= accuracy_score(y_test,y_pred)


# In[10]:


accuracy


# In[11]:


from sklearn.pipeline import make_pipeline
import re
import webbrowser

# Function to preprocess text of the user_input
def preprocess_text(text):
    #Here we put all the features to remove the meaningless text and lower-casing the input of user_input
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = text.lower()
    return text

# Create a pipeline with vectorizer and classifier which we re-use the model features in predicting the intent
model_pipeline = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Chossing column to make the model to predict intent based on user inout 
X = df['cleaned_text']
y = df['intent']

# Fit the pipeline on the entire dataset
model_pipeline.fit(X, y)


# In[12]:


import pyttsx3
import speech_recognition as sr
import datetime
import webbrowser


engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)


def speak(audio):
    engine.say(audio)
    engine.runAndWait()


# speak('Hello, how may I help you')

#Take Command and Converting into text from audio
def takeCommand():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening.....")
        r.pause_threshold = 1
        audio = r.listen(source)
    try:
        print("Recognizing....")
        query = r.recognize_google(audio, language='en-in')
        print(f"user said:{query}\n")
        speak(query)

    except Exception as e:
        # print(e)
        print("say that again please......")
        speak("say that again sir.....")
        return "None"
    return query
# speak('Hello, How can I help you')
# takeCommand()


# In[13]:


import requests
import json

# Assume you have the preprocess_text function and model_pipeline variable defined somewhere in your code.

def predict_intent(query):
    # Preprocess the input text
    preprocessed_text = preprocess_text(query)
    
    # Use the trained pipeline to predict intent
    predicted_intent = model_pipeline.predict([preprocessed_text])[0]   
    return predicted_intent


# In[14]:


#Function for Extracting City Names for weather 

known_city_names=['mumbai','delhi','cheenai','india']

def extract_city_names(query):
    # Split the query into words
    words = query.split()
    # Look for the city name in the query
    for word in words:
        # Check if the word is a city name
        if word.lower() in known_city_names:  
            return word.lower()
    return None


# In[15]:


# function to get news based on user query
#Functionality we are using to make assistant to perform the task!
def get_news(query):
    try:
        url = "http://newsapi.org/v2/top-headlines?country=in&apiKey=3196e5c94880464fbb4f815b17d7c345"
        news_response = requests.get(url)
        news_response.raise_for_status()  

        news = json.loads(news_response.text)
        articles = news.get("articles", [])

        if not articles:
            print("No articles found.")
            speak("Sorry, no articles found.")
            return "No articles found."

        headlines = ""
        for id, article in enumerate(articles[:3], start=1):
            title = article.get('title', '')
            headlines += f"{id}. {title}\n"
        print(headlines)
        speak(headlines)
        return "News fetching complete"

    except requests.RequestException as e:
        print(f"Error fetching news: {e}")
        speak(f"Sorry, there was an error fetching the news. Error: {e}")
        return f"Error fetching news: {e}"


def get_weather(query):
    api_key = 'f03be60b0a6b9e0be78f8d9866f0b663'
    city_name=query
    
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={api_key}"
        weather_response = requests.get(url)
        weather_response.raise_for_status()
        weather = json.loads(weather_response.text)

        # Extracting weather information
        description = weather['weather'][0]['description']
        temperature = weather['main']['temp']
        humidity = weather['main']['humidity']

        response = f"The current weather in {city_name} is {description.lower()} with a temperature of {temperature}°C and humidity of {humidity}%."
        return response
    except requests.RequestException as e:
        print(f"Error fetching weather: {e}")
        return "Sorry, there was an error while fetching the weather."

# function to search Google based on user query
def search_google(query):
    speak('What would you like to search on Google?')
    search_query = takeCommand()
    
    if search_query:
        speak(f'Searching for {search_query} on Google')
        search_url = f'https://www.google.com/search?q={search_query}'
        webbrowser.open(search_url)
    else:
        speak('No search query provided. Please try again.')
        

def search_youtube(query):
    speak('What would you like to search on youtube')
    search_query=takeCommand()
    
    if search_query:
        speak(f'Searching for {search_query} on youtube')
        search_url=f"https://www.youtube.com/search?q={search_query}"
        webbrowser.open(search_url)
    else:
        speak('No search query provided. Please try again.')


# In[16]:


# if __name__ == '__main__':
#     while True:
#         # Get user command
#         query = input('Enter your text: ')
#         weather=query
        
#         # Predict user intent
#         query = predict_intent(query)
#         print(f'Predicted Intent: {query}')

#         # Perform tasks based on the recognized intent
#         if 'search_google' in query:
#             search_google(query)
#         elif 'get_news' in query:
#             speak('News for Today')
#             get_news(query)
#         elif 'search_youtube' in query:
#             search_youtube(query)
#         elif weather:
#             city_name=extract_city_names(weather)
#             if city_name:
#                 weather_info = get_weather(city_name)
#                 print(weather_info)
#                 speak(weather_info)
#             else:
#                 print("Error")
#         elif 'greeting' in query:
#             speak('Hello how may I help you')
#         elif 'joke' in query:
#             speak('You have to fetched API related to what jokes you like to hear')
        
#         else:
#             print('Not found')
        
#         if 'bye' in query:
#             break


# In[17]:


def voice_based_virtual_assistant():
    a='Hello there! Im Your Virtual Assistant ready to assist you. How can I help you today?'
    speak(a)
    print(a)

    while True:
        # Get user command
        query = input('Enter your text: ')
        weather = query

        # Predict user intent
        query = predict_intent(query)
        print(f'Predicted Intent: {query}')

        # Perform tasks based on the recognized intent
        if 'search_google' in query:
            search_google(query)
        elif 'get_news' in query:
            speak('News for Today')
            get_news(query)
        elif 'search_youtube' in query:
            search_youtube(query)
        elif weather:
            city_name = extract_city_names(weather)
            if city_name:
                weather_info = get_weather(city_name)
                print(weather_info)
                speak(weather_info)
            else:
                print("Error")
        elif 'greeting' in query:
            speak('Hello how may I help you')
        elif 'joke' in query:
            speak('You have to fetch API related to what jokes you like to hear')
        elif 'bye' in query:
            break
        else:
            print('Not found')

# Call the function to run the virtual assistant


# # Voice Based Virtual Assistant 

# In[ ]:


if __name__ == '__main__':
    voice_based_virtual_assistant()


# In[ ]:




