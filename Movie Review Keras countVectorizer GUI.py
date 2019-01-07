from tkinter import *
from functools import partial
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

root = Tk()

# create and set variable for the GUI
text = StringVar()
text.set("")
kerasPrediction = IntVar()
kerasAccur = IntVar()
kerasStd = IntVar()

stop_set = set(stopwords.words('english'))

# train data_set that has been modified, example: user said that interesting is a positive word but it predicted wrong
# so user can add the data to review.txt and uncomment this code
# train_dataset = pd.read_csv("data/review.txt", names=['sentiment'], encoding = "ISO-8859-1")

# Read data from file
train_dataset = pd.read_csv("data/imdb_train.txt", names=['sentiment'])  # comment this if you want to use modified data
# test_dataset = pd.read_csv("data/imdb_test.txt", names=['txt'])
train_dataset[['sentiment', 'txt']] = train_dataset["sentiment"].str.split(" ", 1, expand=True)
train_dataset["txt"] = train_dataset["txt"]

# number of training example
batch_size = 10

# SECTION TO ADD STOPSET
stop_set.add("something")
stop_set.add("movie")

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
estimators.append(('mlp', KerasClassifier(build_fn=create_model, epochs=2, batch_size=batch_size)))
pipeline = Pipeline(estimators)

# set kfold parameter to be used to check model accuracy
kfold = StratifiedKFold(n_splits=2, shuffle=True)
result = cross_val_score(pipeline, x_train, y_train, cv=kfold)

# fit the train data
pipeline.fit(x_train, y_train)

kerasAccur.set(result.mean() * 100)
kerasStd.set(result.std() * 100)


def keras_review(pip):
    text.set(input1.get())
    # input to be predicted
    movie_reviews = np.array([input1.get()])

    # map the input with vecotorizer
    x_test = vectorizer.transform(movie_reviews)

    # predict the input data
    predictions = pip.predict(x_test)
    kerasPrediction.set(predictions[0][0])
    word_tokens = word_tokenize(movie_reviews[0])
    filtered_sentence = [w for w in word_tokens if w not in stop_set]

    # print number of number for division and number to be divided
    print(x_test)
    # print the word that occur in sequence
    print(filtered_sentence)


# callable function with argument, will be called by button1
action_with_arg = partial(keras_review, pipeline)

# making frame in the top of the program
topFrame = Frame(root)
topFrame.pack()

# making frame in the bottom of the program
botFrame = Frame(root)
botFrame.pack(side=BOTTOM)

label1 = Label(topFrame, text="INPUT REVIEW: ")
input1 = Entry(topFrame, width="100")
button1 = Button(topFrame, text="Predict!", fg="black", command=action_with_arg)

# putting into the grids
label1.grid(row=0)
input1.grid(row=0, column=1)
button1.grid(row=1, columnspan=2)

# making lables for the accuracy, prediction, and the std in the bottom frame
accurLabel = Label(botFrame, text="Accuracy:")
accurResLabelKeras = Label(botFrame, textvariable=kerasAccur)
predictLabel = Label(botFrame, text="Prediction:")
predictionResLabelKeras = Label(botFrame, textvariable=kerasPrediction)
stdLabelKeras = Label(botFrame, text="Standard Deviation:")
stdResLabelNb = Label(botFrame, textvariable=kerasStd)

# putting into the grids
accurLabel.grid(row=0)
accurResLabelKeras.grid(row=0, column=1)
predictLabel.grid(row=0, column=2)
predictionResLabelKeras.grid(row=0, column=3)
stdLabelKeras.grid(row=0, column=4)
stdResLabelNb.grid(row=0, column=5)

# size of window app
root.geometry("700x150")

# making it not resizeable
root.resizable(0, 0)

# title of the windows(program)
root.title("Keras with CountVectorizer Prediction")

# make sure the app keeps running before it's closed
root.mainloop()
