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

# Data dependencies
import pandas as pd
import numpy as np
# Graphing Libraries
# Visualisation Libraries
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
from nltk import TreebankWordTokenizer

# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")

chart_data = pd.DataFrame(np.random.randn(50),columns=["a"])

# bag of words
def bag_of_words_count(words, word_dict={}):
    """ this function takes in a list of words and returns a dictionary 
        with each word as a key, and the value represents the number of 
        times that word appeared"""
    for word in words:
        if word in word_dict.keys():
            word_dict[word] += 1
        else:
            word_dict[word] = 1
    return word_dict

# tokenise data
tokeniser = TreebankWordTokenizer()

word_dict = {}


# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title("Tweet Classifer")
	st.subheader("Climate change tweet classification")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Prediction", "Information", "Graphics"]
	selection = st.sidebar.selectbox("Choose Option", options)

	# Building out the "Information" page
	if selection == "Information":
		st.info("General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown("Some information here")

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

	# Building out the predication page
	if selection == "Prediction":
		st.info("Prediction with ML Models")
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")
		tokens = tokeniser.tokenize(tweet_text)
		word_dict = bag_of_words_count(tokens)

		if st.button("Classify"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
			prediction = predictor.predict(vect_text)

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			st.success("Text Categorized as: {}".format(prediction))
			word_df = pd.DataFrame(word_dict.items(), index = word_dict.keys())
			#word_df.set_index(0)
			st.bar_chart(word_df)

	# Building out the "Graphics" page
	if selection == "Graphics":
		# You can read a markdown file from supporting resources folder
		#st.markdown("Most Frequently used words")

		st.subheader("Detailed Information about the Tweet")
		#graph = st.radio(
		#"What Information Do You Want To View?",
		#('Number of Words Used', 'Frequently used Words', 'More..'))

		#if graph == 'Word Cloud':
			#wordcloud_ = WordCloud(width=1500, height = 900).generate(raw['message'][0])
			# Display the generated image:
			#plt.imshow(wordcloud_, interpolation='bilinear')
			#plt.set_title('Words in Tweet')
			#plt.axis('off')
			#st.pyplot()
            # Show the ditribution of the classes as a graph
		#elif graph == 'Word Count':
			#f, ax = plt.subplots(figsize=(10, 8))
			#sns.set(style="whitegrid")
			#ax = sns.countplot(x="message", data=raw['message'][0])
			#plt.title('Message Count', fontsize =20)
			#st.pyplot()
            
		#elif graph == 'Number of Words Used':
		st.bar_chart(word_dict)

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
