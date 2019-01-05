# UNCOMMENT IMPORT ONLY IF NEEDED
# import nltk
# nltk.download('stopwords')
import pandas as pd
from nltk.corpus import stopwords
from sklearn import naive_bayes
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
import numpy as np

# Sentiment -> positive = 1; negative = 0
# to separate the class and the features
train_dataset = pd.read_csv("data/imdb_train.txt", names=['sentiment'])
test_dataset = pd.read_csv("data/imdb_test.txt", names=['txt'])
train_dataset[['sentiment', 'txt']] = train_dataset["sentiment"].str.split(" ", 1, expand=True)
train_dataset["txt"] = train_dataset["txt"]

# To remove the common word and count the probability of each word while also doing smoothing
stopset = set(stopwords.words('english'))
vectorizer = TfidfVectorizer(use_idf=True, smooth_idf=True, lowercase=True, strip_accents='ascii', stop_words=stopset)

# assign the value train feature, train target, and test feature
y_train = train_dataset.sentiment
x_train = vectorizer.fit_transform(train_dataset.txt)
x_test = vectorizer.transform(test_dataset.txt)

# using the multinomial naive bayes
clf = naive_bayes.MultinomialNB()
clf.fit(x_train, y_train)
y_test = clf.predict(x_test)

# score = roc_auc_score(y_test, clf.predict_proba(x_test)[:, 1])
# score is 1.0 accuracy, not really good causing overfitting
score = clf.score(x_test, y_test)
print(score)

# cannot see unforeseen data over-fitting, false prediction
movie_reviews = np.array(["Jupiter Ascending was a disappointing and terrible movie",
                          "something is a nice movie"])
movie_review_vector = vectorizer.transform(movie_reviews)
print(clf.predict(movie_review_vector))
