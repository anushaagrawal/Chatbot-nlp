#!/usr/bin/env python
# coding: utf-8

# In[1]:


import io
import random
import string
import warnings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')
import os


# In[2]:


import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('popular', quiet=True)


# In[10]:


nltk.download('punkt') # first-time use only
nltk.download('wordnet') # first-time use only


# In[3]:


with open('jaipur.txt','r', encoding='utf8', errors ='ignore') as fin:
    raw = fin.read().lower()
sent_tokens = nltk.sent_tokenize(raw)# converts to list of sentences 
word_tokens = nltk.word_tokenize(raw)# converts to list of words


# In[4]:


lemmer = WordNetLemmatizer()
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


# In[5]:


GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey",)
GREETING_RESPONSES = ["hi", "hey", "hi there", "hello", "I am glad! You are talking to me"]
def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


# In[6]:


def response(user_response):
    robo_response=''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        robo_response=robo_response+"I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response = robo_response+sent_tokens[idx]
        return robo_response


# In[7]:


#CHATBOT
flag=True
print("ROBO: My name is Robo. I will answer your queries about Jaipur. If you want to exit, type Bye!")
while(flag==True):
    user_response = input()
    user_response=user_response.lower()
    if(user_response!='bye'):
        if(user_response=='thanks' or user_response=='thank you' ):
            flag=False
            print("ROBO: You are welcome..")
        else:
            greet = greeting(user_response)
            if(greet!=None):
                print("ROBO: "+greet)
            else:
                reply = response(user_response)
                print("ROBO: ",end="")
                print(reply)
                sent_tokens.remove(user_response)
    else:
        flag=False
        print("ROBO: Bye! take care..")  
    print()


# In[ ]:





# In[ ]:


from flask import Flask, render_template, request  
#Flask:Used to create a server gateway
#render_template: Used to import the html page and display it in the basic route
#request: Used to get user-input from the website using 'request.args.get()'

#Creating an instance of the class
app = Flask(__name__)     

#Displaying the webpage on the basic route
@app.route("/")
def basic():
    return render_template("chatbot.html")

#Function on the get server to work on the user-input and return the bots reply
@app.route("/get")
def get_bot_response():
    user_response = request.args.get("msg")   
    user_response=user_response.lower()
    if(user_response!='bye'):
        if(user_response=='thanks' or user_response=='thank you' ):
            flag=False
            print("ROBO: You are welcome..")
            return str("You are welcome..")
        else:
            greet = greeting(user_response)
            if(greet!=None):
                print("ROBO: "+greet)
                return greet
            else:
                reply = response(user_response)
                print("ROBO: ",end="")
                print(reply)
                sent_tokens.remove(user_response)
                return reply
    else:
        flag=False
        print("ROBO: Bye! take care..")  
        return str("Bye! take care..")


if __name__ == "__main__":
    app.run(debug = False, host = '0.0.0.0')

