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

stop_words = set(stopwords.words('english'))
filtered_sentence = []

train_dataset = pd.read_csv("data/imdb_train.txt", names=['sentiment'])
# test_dataset = pd.read_csv("data/imdb_test.txt", names=['txt'])
train_dataset[['sentiment', 'txt']] = train_dataset["sentiment"].str.split(" ", 1, expand=True)
train_dataset["txt"] = train_dataset["txt"]

vocab_size = 3000
batch_size = 10

# define Tokenizer with Vocab Size
tokenizer = Tokenizer(num_words=vocab_size, lower=True)

for review_sent in train_dataset.txt:
    filtered_sentence_review = ""
    word_tokens = word_tokenize(review_sent)
    for w in word_tokens:
        if w not in stop_words and w != "'s" and w != "'ve" and w != "n't":
            filtered_sentence_review += w
            filtered_sentence_review += " "
    filtered_sentence.append(filtered_sentence_review)

tokenizer.fit_on_texts(train_dataset.txt)

x_train = tokenizer.texts_to_matrix(train_dataset.txt)
# x_test = tokenizer.texts_to_matrix(test_dataset.txt)
y_train = train_dataset.sentiment

print(x_train)
def create_model():
    model = Sequential()
    model.add(
        Dense(len(x_train[0]), input_dim=len(x_train[0]), kernel_initializer='normal',
              activation='relu'))  # relu to take out any negative value
    model.add(Dense(30, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_model, epochs=5, batch_size=batch_size)))
pipeline = Pipeline(estimators)

kfold = StratifiedKFold(n_splits=2, shuffle=True)
result = cross_val_score(pipeline, x_train, y_train, cv=kfold)

movie_reviews = np.array(["this movie is very good",
                          "this movie is incredible",
                          "this movie is great"])

x_test = tokenizer.texts_to_matrix(movie_reviews)
print(x_test)
pipeline = pipeline.fit(x_train, y_train)
predictions = pipeline.predict(x_test)

print("Results: %.2f% % (%.2f% %)" % (result.mean() * 100, result.std() * 100))

number = 1

for prediction in predictions:
    print("Prediction of review ", number, " =", prediction)
    number += 1
