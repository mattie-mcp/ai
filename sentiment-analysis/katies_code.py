
# Team Asimov's Laws: Katie Mulder, Ben LaFeldt, Mattie Phillips
# 
# 
# With an 80/20 split, the following code resulted in a 90.03% accuracy
# on the data provided. The GridSearchCV selected the following paramters:
#
# The following lines were modified, with comments preceeding them.
# Lines Modified: 35-74, 81-82, 90, 100, 104, 108-114

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
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop = stopwords.words('english')
 
print("Execution Started.")

# Below are several tokenization strategies tried.
# The default tokenization strategy yielded the highest accuracy.
porter = PorterStemmer()
		
# Remove only HTML tags from the text.
def tokenizer(text):
    text2 = re.sub("<.*>", " ", text)
    return text2.split()

# Remove most punctuation marks and HTML tags from the text.
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

# Remove HTML tags and implment the tokenization using porter stemming.
def tokenizer_porter_no_html(text):
    text2 = re.sub("<.*>", "", text)
    ary = []
    for word in text4.split():
        if len(word) > 2:
            if word != "OED":
                ary.append( porter.stem(word))
    return ary

# Remove most punctuation and implement tokenization using porter stemming.
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

# Randomize the data
np.random.seed()
df = df.reindex(np.random.permutation(df.index))

# Split your data into training and test sets.
# Allows you to train the model, and then perform
# validation to get a sense of performance.

# An 80/20  split of the data gave the most accurate split of data.
# Other splits tried: 75/25, 70/30, and 90/10. 
training_size = 45000*.8
X_train = df.loc[:training_size, 'review'].values
y_train = df.loc[:training_size, 'sentiment'].values
X_test = df.loc[training_size:, 'review'].values
y_test = df.loc[training_size:, 'sentiment'].values

# Perform feature extraction on the text.
print("Creating Tokenizer.")
# The parameters used below were parameters consistenly selected by the GridSearchCV, removed from the parameter grid
# to limit the computation time required.
tfidf = TfidfVectorizer(strip_accents='unicode', stop_words=None, tokenizer=None, lowercase=False, preprocessor=None)

# Another classifier that we tried was the SGDClassifier, but the LogisticRegression yielded the most accurate results.
print("Starting pipeline and training")
lr_tfidf = Pipeline([('vect', tfidf), ('clf', LogisticRegression(0))])

# The below parameters yielded the most accuracy. 
# Other Parameters explored: vect__min_df, vect__tokenizer, vect__stop_words, vect_lowercase, min_df, among others. 
param_grid = [{
		'vect__max_df': [.85, .9],
              	'clf__penalty': ['l2'],
		'clf__C': [5.0, 5.5, 6.0, 6.5]
	     }]
gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid, scoring='accuracy', cv=10, verbose=1, n_jobs=-1)
gs_lr_tfidf.fit(X_train, y_train)

# Print the Test Accuracy
print('CV Accuracy: %.8f' % gs_lr_tfidf.best_score_)

clf = gs_lr_tfidf.best_estimator_
print('Test Accuracy: %.8f' % clf.score(X_test, y_test))

print (gs_lr_tfidf.best_params_)
      
# Save the classifier for use later.
pickle.dump(gs_lr_tfidf, open("saved_model.sav", 'wb'))

print("Execution finished.")




