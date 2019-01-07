import pandas as pd
from keras_preprocessing.text import Tokenizer
from sklearn.pipeline import Pipeline
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
import numpy as np
# You can comment after running once
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# bag of unnecessary word
stop_words = set(stopwords.words('english'))
filtered_sentence = []

# train data_set that has been modified, example: user said that interesting is a positive word but it predicted wrong
# so user can add the data to review.txt and uncomment this code
# train_dataset = pd.read_csv("data/review.txt", names=['sentiment'], encoding = "ISO-8859-1")

# read data from files
train_dataset = pd.read_csv("data/imdb_train.txt", names=['sentiment'])  # comment this if you want to use modified data
train_dataset[['sentiment', 'txt']] = train_dataset["sentiment"].str.split(" ", 1, expand=True)
train_dataset["txt"] = train_dataset["txt"]

# number of top 3000 vocabulary to be used
vocab_size = 3000
# number of training data
batch_size = 10

# define Tokenizer with top 3000 vocab
tokenizer = Tokenizer(num_words=vocab_size, lower=True)

# removing unnecessary words from the review
for review_sent in train_dataset.txt:
    filtered_sentence_review = ""
    word_tokens = word_tokenize(review_sent)
    for w in word_tokens:
        if w not in stop_words and w != "'s" and w != "'ve" and w != "n't":
            filtered_sentence_review += w
            filtered_sentence_review += " "
    filtered_sentence.append(filtered_sentence_review)

# fit the data to tokenizer
tokenizer.fit_on_texts(train_dataset.txt)

# convert tokenizer to matrix
x_train = tokenizer.texts_to_matrix(train_dataset.txt)
y_train = train_dataset.sentiment

# create model to be used for the classifier
def create_model():
    model = Sequential()
    model.add(
        Dense(len(x_train[0]), input_dim=len(x_train[0]), kernel_initializer='normal',
              activation='relu'))  # relu to take out any negative value
    model.add(Dense(30, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# pipeline to run several process
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_model, epochs=5, batch_size=batch_size)))
pipeline = Pipeline(estimators)

# define the kfold to be used to check the model accuracy
kfold = StratifiedKFold(n_splits=2, shuffle=True)
# store the result of the model accuracy
result = cross_val_score(pipeline, x_train, y_train, cv=kfold)

# print the model accuracy
print("Results: %.2f% % (%.2f% %)" % (result.mean() * 100, result.std() * 100))

# input to be predicted
movie_reviews = np.array(["this movie is very good",
                          "this movie is incredible",
                          "this movie is great"])

# tokenize, fit, and predict the input
x_test = tokenizer.texts_to_matrix(movie_reviews)
pipeline = pipeline.fit(x_train, y_train)
predictions = pipeline.predict(x_test)

#print prediction of the input
number = 1

for prediction in predictions:
    print("Prediction of review ", number, " =", prediction)
    number += 1
