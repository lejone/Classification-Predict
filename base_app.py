"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Plase follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os
import re
from collections import Counter

# Visualisation Libraries

import seaborn as sns
import altair as alt
from vega_datasets import data #pip install vega_datasets

# Data dependencies
import pandas as pd
import numpy as np
import pickle
import nltk
from nltk.tokenize import word_tokenize, TreebankWordTokenizer 
from nltk.stem import WordNetLemmatizer 
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import spacy
#spacy.cli.download('en_core_web_sm')
#import spacy.cli

#ML libraries
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer

# Load your raw data
raw = pd.read_csv("resources/train.csv")
# Load preprocessed data
pre= pd.read_csv("resources/preprocessed.csv")

# Load randomforest and logistic pickles
model_load_path = "resources/RF.pkl"
with open(model_load_path,'rb') as file:
    Rf_model = pickle.load(file)

model_load_path = "resources/log_reg.pkl"
with open(model_load_path,'rb') as file:
    log_model = pickle.load(file)


# The main function where we will build the actual app
def main():


	#"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title("Tweet Cl@ssifer")

	st.subheader("Climate Change Tweet Sentiment Classification")
	
	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Insights","Prediction"]
	selection = st.sidebar.selectbox("Choose Option", options)

	# Building out the "Information" page

	if selection == "Insights":
		#st.info("General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown("HAVE A LOOK AT SOME INSIGHTS FROM OUR TWEETS!!")

		st.subheader("Tweets and Sentiment")
		if st.checkbox('View Raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

		st.subheader("Who is Trending?")
		if st.checkbox('View People'): # data is hidden if box is unchecked

		#Retrive all the PERSON labels from the pre dataframe and generate a new dataframe person_df for analysis
			
			persons=[i for i in pre['persons']]
			person_counts = Counter(persons).most_common(20)
			person_df=pd.DataFrame(person_counts,columns=['Persons Name','Mentions'])
	
			person_df.drop([0,1,7,8,10,11,17],axis=0,inplace=True) # rows removed due to 'None' entries, incorrect classification or different entry of a same entity (repetition)  
			p=alt.Chart(person_df).mark_bar(size=30).encode(x='Persons Name',y='Mentions',color=alt.Color('Persons Name', scale=alt.Scale(scheme='category20'))).properties(width=700,height=500)
			st.altair_chart(p)

		st.subheader("Which Institutions are Trending?")
		if st.checkbox('View Institutions'): # data is hidden if box is unchecked
		
		#Retrive all the ORG labels from the NER_df and generate a new dataframe org_df for analysis
			orgs=[i for i in pre['organisation']]

		#plot top organisations tweeted
			org_counts = Counter(orgs).most_common(20)
			org_df=pd.DataFrame(org_counts,columns=['Institution Name','Mentions'])
			org_df.drop([0,1,2,4,9],axis=0,inplace=True) # rows removed due to 'None' entries, incorrect classification or different entry of a same entity (repetition)  
			
			o=alt.Chart(org_df).mark_bar(size=30).encode(x='Institution Name',y='Mentions',color=alt.Color('Institution Name', scale=alt.Scale(scheme='category20'))).properties(width=700,height=500)

			st.altair_chart(o)
	
	# Building out the predication page

	if selection == "Prediction":
		st.markdown("Sentiment Predictions : What your stance?")
		st.markdown("*[-1] A global warming sceptic*")
		st.markdown("*[0] meh*")
		st.markdown("*[1] Environmentally Conscious*")
		st.markdown("*[2] Tweet like a reporter*")
		st.markdown("LETS FIND OUT!! :sunglasses:")
		st.markdown("CLASSIFY YOUR TWEET!!")

		# Creating a text box for user input

		tweet_text = st.text_area("Enter Text","Type Here")

		# Generate tweet and vader dataframes

		def dframe(text):
			dic={'message':text}
			df=pd.DataFrame(dic,index=[0])
			return df

		tweet=dframe(tweet_text)
		vader=dframe(tweet_text)

		# Function to remove/replace unwanted text such as characters,URLs etc
		
		def clean(text):
			text=text.replace("'",'')
			text=text.replace(".",' ')
			text=text.replace("  ",'')
			text=text.replace(",",' ')
			text=text.replace("_",' ')
			text=text.replace("!",' ')
			text=text.replace("RT",'retweet') #Replace RT(Retweet) with relay
			text=text.replace(r'\d+','')
			text=re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(https?//[^\s]+))','weblink',text)
			text=re.sub('((co/[^\s]+)|(co?://[^\s]+)|(co?//[^\s]+))','',text)
			text=text.lower()  # Lowercase tweet
			text =text.lstrip('\'"') # Remove extra white space
			
			return text

		tweet['message']=tweet['message'].apply(clean)

		
		#Function removes punctuation

		def rm_punc(text):
			clean_text=[]
			for i in str(text).split():
				rm=i.strip('\'"?,.:_/<>!')
				clean_text.append(rm)
			return ' '.join(clean_text)

		tweet['message']=tweet['message'].apply(rm_punc)

		# Function replaces the @ symbol with the word at
		
		def at(text):
			return ' '.join(re.sub("(@+)","at ",text).split())

		tweet['message']=tweet['message'].apply(at)

		# Functions replaces the # symbol with the word tag
		
		def hashtag(text):
			return ' '.join(re.sub("(#+)"," tag ",text).split())

		tweet['message']=tweet['message'].apply(hashtag)

		# Tokenise each tweet messge

		tokeniser = TreebankWordTokenizer()
		tweet['tokens'] = tweet['message'].apply(tokeniser.tokenize)

		# Function lemmatizes text
			
		def lemma(text):
			lemma = WordNetLemmatizer() 
			return [lemma.lemmatize(i) for i in text]

		tweet['lemma']=tweet['tokens'].apply(lemma)

		tweet['clean message'] = tweet['lemma'].apply(lambda i: ' '.join(i))

		# Classification options to chose from

		clf_options = ["VADER Sentiment Analysis","Random Forest Classification","Logistic Regression"]
		clf_selection = st.radio("Choose Option", clf_options)

		# Vader sentiment analysis
		# Generate compound scores for VADER analysis

		sid = SentimentIntensityAnalyzer()

		if clf_selection == "VADER Sentiment Analysis":
			st.write("You have chosen to use VADER Sentiment analysis")
			st.info("VADER is not machine learning model but a lexicon that determines your sentiment based off how positive or negative each of the words you used are")
			st.markdown("*Is your sentiment positive or negative?*")
			sid = SentimentIntensityAnalyzer()
			vader['scores'] = vader['message'].apply(lambda i: sid.polarity_scores(i))
			vader['compound']  = vader['scores'].apply(lambda score_dict: score_dict['compound'])
			vader['sentiment'] = vader['compound'].apply(lambda i: 'POSITIVE' if i >=0 else 'NEGATIVE')
			if st.button("Classify"):
				
				st.success("Text Categorized as: {}".format(vader["sentiment"][0]))

		# Build randomforest classification

		if clf_selection == "Random Forest Classification":
			st.write("You have chosen to use a Random Forest Classifier")
			st.info("This classifier combines many decision trees into a single model")
			if st.button("Classify"):
				rf_pred=Rf_model.predict(tweet['clean message'])
				st.success("Text Categorized as: {}".format(rf_pred))

		# Build logistic regression

		if clf_selection == "Logistic Regression":
			st.write("You have chosen to use a Logistic Regression Classifier")
			st.info("This model predicts binary outcomes based of given independent variables")
			if st.button("Classify"):
				log_pred=log_model.predict(tweet['clean message'])
				st.success("Text Categorized as: {}".format(log_pred))

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
