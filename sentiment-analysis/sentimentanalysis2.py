
# coding: utf-8

# In[ ]:

"""
    Train a logistic regresion model for document classification.

    Search this file for the keyword "Hint" for possible areas of
    improvement.  There are of course others.
"""
print("Execution Started.")

import pandas as pd
import pickle
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import sys
import re
import string
from sklearn.linear_model import SGDClassifier

# Hint: These are not actually used in the current 
# pipeline, but would be used in an alternative 
# tokenizer such as PorterStemming.
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop = stopwords.words('english')

"""
    This is a very basic tokenization strategy.  
    
    Hint: Perhaps implement others such as PorterStemming
    Hint: Is this even used?  Where would you place it?
"""

porter = PorterStemmer()

def tokenizer(text):
    #text2 = re.sub("<.*>", "", text)
    return text.split()
		
def tokenizer2(text):
    text2 = re.sub("<.*>", "", text)
    return text2.split()

def tokenizer_porter(text):
    text2 = re.sub("<.*>", "", text)
    text3 = re.sub('[/?/."!:<>(-)@#$%~&/*+,]*', "", text2)
    text4 = re.sub("['*]*", "", text3)
    ary = []
    for word in text4.split():
        if len(word) > 2:
            if word != "OED":
                ary.append( porter.stem(word))
    return ary

def tokenizer_porter_no_punct_html(text):
    text2 = re.sub("<.*>", "", text)
    text3 = re.sub('[/?/."!:<>(-)@#$%~&/*+,]*', "", text2)
    text4 = re.sub("['*]*", "", text3)
    ary = []
    for word in text4.split():
        if len(word) > 2:
            if word != "OED":
                ary.append( porter.stem(word))
    return ary


# Read in the dataset and store in a pandas dataframe
df = pd.read_csv('./training_movie_data.csv')


#Randomize the data
np.random.seed()
df = df.reindex(np.random.permutation(df.index))

# Split your data into training and test sets.
# Allows you to train the model, and then perform
# validation to get a sense of performance.
# 
# Hint: This might be an area to change the size
# of your training and test sets for improved 
# predictive performance.
training_size = 45000*.8
X_train = df.loc[:training_size, 'review'].values
y_train = df.loc[:training_size, 'sentiment'].values
X_test = df.loc[training_size:, 'review'].values
y_test = df.loc[training_size:, 'sentiment'].values

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV

# Perform feature extraction on the text.
# Hint: Perhaps there are different preprocessors to test? 
print("Starting Feature Extraction.")
tfidf = TfidfVectorizer(strip_accents='unicode',
						stop_words=None,
						tokenizer=None,
                        lowercase=False,
                        preprocessor=None)

# Hint: There are methods to perform parameter sweeps to find the
# best combination of parameters.  Look towards GridSearchCV in 
# sklearn or other model selection strategies.

# Create a pipeline to vectorize the data and then perform regression.
# Hint: Are there other options to add to this process?
# Look to documentation on Regression or similar methods for hints.
# Possibly investigate alternative classifiers for text/sentiment.
print("Starting pipeline and training")
lr_tfidf = Pipeline([('vect', tfidf),
                     ('clf', LogisticRegression())
					 #('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42)),
])

param_grid = [{
				'vect__stop_words': [None],
				'vect__strip_accents': [None, 'ascii', 'unicode'],
				'vect__tokenizer': [None],
				'vect__lowercase': [False],
				'vect__preprocessor': [None],
				'vect__lowercase': [False],
				'vect__max_df': [.8, .9, .95, 1.0],
				'vect__min_df': [0.0, .05, .01, .1], 
         		#'vect__tokenizer': [None, tokenizer, tokenizer2],
				#'vect__tokenizer': [None, tokenizer, tokenizer_porter_no_punct_html],
              	#'clf__penalty': ['l1', 'l2', '13'],
              	'clf__penalty': ['l1', 'l2'],
				#'clf__C': [1.0, 10.0, 100.0]
				#'clf__C': [10.0]
              	#'clf__C': [4.0, 5.0, 6.0, 9.0, 10.0, 11.0, 15.0, 100.0]
				'clf__C': [4.0, 5.0, 6.0, 9.0, 10.0, 11.0, 15.0]
				
				#'clf_loss':['hinge'], 
				#'clf_penalty':['l2'], 
				#'clf__alpha':[.001, .0001], 
				#'clf__n_iter':[1,5,10] 
			}]

gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid,
                           scoring='accuracy',
							cv=5,
                           #cv=10,
                           verbose=10,
                           n_jobs=-1)

gs_lr_tfidf.fit(X_train, y_train)


# Print the Test Accuracy
print('CV Accuracy: %.8f' % gs_lr_tfidf.best_score_)

clf = gs_lr_tfidf.best_estimator_
print('Test Accuracy: %.8f' % clf.score(X_test, y_test))

print (gs_lr_tfidf.best_params_)
      
# Save the classifier for use later.
pickle.dump(gs_lr_tfidf, open("saved_model.sav", 'wb'))

print("##########################################\n\n\nExecution finished.")

      


# In[ ]:



