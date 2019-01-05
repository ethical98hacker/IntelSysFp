# UNCOMMENT IMPORT ONLY IF NEEDED
# import nltk
# nltk.download('stopwords')
import pandas as pd
from nltk.corpus import stopwords
from sklearn import naive_bayes
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Sentiment -> positive = 1; negative = 0
# to separate the class and the features
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold

train_dataset = pd.read_csv("data/imdb_train.txt", names=['sentiment'])
predict_dataset = pd.read_csv("data/imdb_test.txt", names=['txt'])
train_dataset[['sentiment', 'txt']] = train_dataset["sentiment"].str.split(" ", 1, expand=True)
train_dataset["txt"] = train_dataset["txt"]

# To remove the common word and count the probability of each word while also doing smoothing
stopset = set(stopwords.words('english'))
vectorizer = TfidfVectorizer(use_idf=True, smooth_idf=True, lowercase=True, strip_accents='ascii', stop_words=stopset)

x = vectorizer.fit_transform(train_dataset.txt)
y = train_dataset.sentiment

# x_predict = vectorizer.transform(predict_dataset)
# print(x_predict)

# assign the value train feature, train target, and test feature
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=50)

# using the multinomial naive bayes
clf = naive_bayes.MultinomialNB()
clf.fit(x_train, y_train)

# To see whether the model is overfitting or underfitting
kfold = StratifiedKFold(n_splits=2, shuffle=True)
model_cv_score = cross_val_score(estimator=clf, X=x, y=y, cv=kfold, verbose=True)
print("Results: %.2f% % (%.2f% %)" % (model_cv_score.mean() * 100, model_cv_score.std() * 100))

movie_reviews = np.array(["it was a terrible movie",
                          "this is a nice movie",
                          "check out this movie, it is a great movie.. worth to watch"])

movie_review_vector = vectorizer.transform(movie_reviews)
print(movie_review_vector)
print(clf.predict(movie_review_vector))
