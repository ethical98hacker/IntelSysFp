from tkinter import *
from functools import partial
# downloads bags of words for words list (you can comment it if you have already run the program once)
# import nltk
# nltk.download('stopwords')
import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn import naive_bayes
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold

root = Tk()

# create and set variable for the GUI
text = StringVar()
text.set("")
nbPrediction = IntVar()
nbAccur = IntVar()
nbStd = IntVar()

# train data_set that has been modified, example: user said that interesting is a positive word but it predicted wrong
# so user can add the data to review.txt and uncomment this code
train_dataset = pd.read_csv("data/review.txt", names=['sentiment'], encoding = "ISO-8859-1")

# NAIVE BAYES
# Sentiment -> positive = 1; negative = 0
# to separate the class and the features
# train_dataset = pd.read_csv("data/imdb_train.txt", names=['sentiment'])  # comment this if you want to use modified data
predict_dataset = pd.read_csv("data/imdb_test.txt", names=['txt'])
train_dataset[['sentiment', 'txt']] = train_dataset["sentiment"].str.split(" ", 1, expand=True)
train_dataset["txt"] = train_dataset["txt"]


# Showing the data of positive and negative review distribution are equal
def showplot(train_data):
    train_data.groupby('sentiment').txt.count().plot.bar(ylim=0)
    plt.show()


# callable function with argument, will be called by button2
action_with_arg2 = partial(showplot, train_dataset)

# bags of common word
stopset = set(stopwords.words('english'))

# SECTION TO ADD STOPSET
stopset.add("something")

# set the vectorizer
vectorizer = TfidfVectorizer(use_idf=True, smooth_idf=True, lowercase=True, strip_accents='ascii', stop_words=stopset,
                             ngram_range=(1, 1))

# map the train word using the vectorizer
x = vectorizer.fit_transform(train_dataset.txt)
y = train_dataset.sentiment

# divide data -> train data: 60%; test data: 40% randomly
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=50)

# using the multinomial naive bayes
clf = naive_bayes.MultinomialNB()
clf.fit(x_train, y_train)

# check model: under-fitting or over-fitting
kfold = StratifiedKFold(n_splits=2, shuffle=True)
model_cv_score = cross_val_score(estimator=clf, X=x, y=y, cv=kfold, verbose=True)
print("Results: %.2f% % (%.2f% %)" % (model_cv_score.mean() * 100, model_cv_score.std() * 100))
nbAccur.set(model_cv_score.mean() * 100)
nbStd.set(model_cv_score.std() * 100)


# function to predict user input after button1 pressed
def NaiveBayes(v):
    text.set(input1.get())
    movie_reviews = np.array([input1.get()])
    movie_review_vector = v.transform(movie_reviews)
    nbPrediction.set(clf.predict(movie_review_vector)[0])
    word_tokens = word_tokenize(movie_reviews[0])
    filtered_sentence = [w for w in word_tokens if w not in stopset]

    # print the probability of each word
    print(movie_review_vector)
    # list of the word in sequence
    print(filtered_sentence)


# callable function with argument, will be called by button1
action_with_arg = partial(NaiveBayes, vectorizer)

# making frame in the top of the program
topFrame = Frame(root)
topFrame.pack()

# making frame in the bottom of the program
botFrame = Frame(root)
botFrame.pack(side=BOTTOM)

# making labels for the text input in the top frame
label1 = Label(topFrame, text="INPUT REVIEW: ")
input1 = Entry(topFrame, width="100")
button1 = Button(topFrame, text="Predict!", fg="black", command=action_with_arg)

# putting into the grids
label1.grid(row=0)
input1.grid(row=0, column=1)
button1.grid(row=1, columnspan=2)

# making lables for the accuracy, prediction, and the std in the bottom frame
accurLabel = Label(botFrame, text="Accuracy:")
accurResLabelNb = Label(botFrame, textvariable=nbAccur)
predictLabel = Label(botFrame, text="Prediction:")
predictionResLabelNb = Label(botFrame, textvariable=nbPrediction)
stdLabelNb = Label(botFrame, text="Standard Deviation:")
stdResLabelNb = Label(botFrame, textvariable=nbStd)
button2 = Button(botFrame, text="Show Data Distribution", fg="black", command=action_with_arg2)

# putting into the grids
accurLabel.grid(row=0)
accurResLabelNb.grid(row=0, column=1)
predictLabel.grid(row=0, column=2)
predictionResLabelNb.grid(row=0, column=3)
stdLabelNb.grid(row=0, column=4)
stdResLabelNb.grid(row=0, column=5)
button2.grid(row=1, columnspan=10)

# size of window app
root.geometry("700x150")

# making it not resizeable
root.resizable(0, 0)

# title of the windows(program)
root.title("Naive Bayes Prediction")

# make sure the app keeps running before it's closed
root.mainloop()
