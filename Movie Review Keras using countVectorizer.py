import pandas as pd
from nltk import word_tokenize
from sklearn.pipeline import Pipeline
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# You can comment after running once
# import nltk
# nltk.download('stopwords')
# nltk.download('punkt')

from nltk.corpus import stopwords

stop_set = set(stopwords.words('english'))

# Belom ada smoothing and n-gram
train_dataset = pd.read_csv("data/imdb_train.txt", names=['sentiment'])
# test_dataset = pd.read_csv("data/imdb_test.txt", names=['txt'])
train_dataset[['sentiment', 'txt']] = train_dataset["sentiment"].str.split(" ", 1, expand=True)
train_dataset["txt"] = train_dataset["txt"]

# number of training example
batch_size = 10

# SECTION TO ADD STOPSET
stop_set.add("something")

# set vectorizer to map the word
vectorizer = CountVectorizer(max_df=35000, min_df=500, strip_accents='ascii', lowercase=True, stop_words=stop_set,
                             ngram_range=(1, 1))

# fit the train data
vectorizer.fit(train_dataset.txt)

# map the word after the vectorizer is fitted
x_train = vectorizer.transform(train_dataset.txt).toarray()
y_train = train_dataset.sentiment


# create model to be passed to the classifier
def create_model():
    model = Sequential()
    model.add(
        # input layer
        Dense(len(x_train[0]), input_dim=len(x_train[0]), kernel_initializer='normal',
              activation='relu'))  # relu to take out any negative value
    # hidden layer
    model.add(Dense(30, kernel_initializer='normal', activation='relu'))
    # output layer
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# pipeline to run the classifier
estimators = []
estimators.append(('mlp', KerasClassifier(build_fn=create_model, epochs=5, batch_size=batch_size)))
pipeline = Pipeline(estimators)

# set kfold parameter to be used to check model accuracy
kfold = StratifiedKFold(n_splits=2, shuffle=True)
result = cross_val_score(pipeline, x_train, y_train, cv=kfold)

# input to be predicted
movie_reviews = np.array(["something is very good"])

# map the input with vecotorizer
x_test = vectorizer.transform(movie_reviews)

# fit the train data
pipeline = pipeline.fit(x_train, y_train)
# predict the input data
predictions = pipeline.predict(x_test)

# print accuracy result
print("Results: %.2f% % (%.2f% %)" % (result.mean() * 100, result.std() * 100))

# print prediction of the review
number = 1

# print predictions
for prediction in predictions:
    print("Prediction of review ", number, " =", prediction)
    number += 1

# print the weight of the input data
print(x_test)
word_tokens = word_tokenize(movie_reviews[0])
filtered_sentence = [w for w in word_tokens if w not in stop_set]
# print the words that has been weighted in sequence
print(filtered_sentence)
