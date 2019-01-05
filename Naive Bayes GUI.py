from tkinter import *
from functools import partial
#downloads bags of words for words list (you can comment it if you have already run the program once)
#import nltk
#nltk.download('stopwords')
import pandas as pd
from nltk.corpus import stopwords
from sklearn import naive_bayes
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
root = Tk()

text = StringVar()
text.set("")
nbPrediction = IntVar()
nbAccur = IntVar()
nbStd = IntVar()

#NAIVE BAYES
# Sentiment -> positive = 1; negative = 0
# to separate the class and the features
train_dataset = pd.read_csv("data/imdb_train.txt", names=['sentiment'])
predict_dataset = pd.read_csv("data/imdb_test.txt", names=['txt'])
train_dataset[['sentiment', 'txt']] = train_dataset["sentiment"].str.split(" ", 1, expand=True)
train_dataset["txt"] = train_dataset["txt"]

# Showing the data of positive and negative review distribution are equal
def showplot(train_data):
    train_data.groupby('sentiment').txt.count().plot.bar(ylim=0)
    plt.show()
action_with_arg2 = partial(showplot, train_dataset)

# To remove the common word and count the probability of each word while also doing smoothing
stopset = set(stopwords.words('english'))
vectorizer = TfidfVectorizer(use_idf=True, smooth_idf=True, lowercase=True, strip_accents='ascii', stop_words=stopset,
                             ngram_range=(1, 2))
# convert train data using vectorizer
x = vectorizer.fit_transform(train_dataset.txt)
y = train_dataset.sentiment

# assign the value train feature, train target, and test feature
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=50)

# using the multinomial naive bayes
clf = naive_bayes.MultinomialNB()
clf.fit(x_train, y_train)

# To see whether the model is overfitting or underfitting
kfold = StratifiedKFold(n_splits=2, shuffle=True)
model_cv_score = cross_val_score(estimator=clf, X=x, y=y, cv=kfold, verbose=True)
print("Results: %.2f% % (%.2f% %)" % (model_cv_score.mean() * 100, model_cv_score.std() * 100))
nbAccur.set(model_cv_score.mean() * 100)
nbStd.set(model_cv_score.std() * 100)

score = clf.score(x_test, y_test)
print("Score :", score)

#the naive bayes function for predicting each input after you click predict button
def NaiveBayes(v):
    text.set(input1.get())
    print ("ini",input1.get())
    movie_reviews = np.array([input1.get()])
    print ("mrv",movie_reviews)
    movie_review_vector = v.transform(movie_reviews)
    nbPrediction.set(clf.predict(movie_review_vector)[0])

action_with_arg = partial(NaiveBayes, vectorizer)

#making frame in the top of the program
topFrame = Frame(root)
topFrame.pack()

#making frame in the bottom of the program
botFrame = Frame(root)
botFrame.pack(side=BOTTOM)

#making labels for the text input in the top frame
label1 = Label(topFrame, text="INPUT REVIEW: ")
input1 = Entry(topFrame, width="100")
button1 = Button(topFrame, text="Predict!", fg="black", command=action_with_arg)

#putting into the grids
label1.grid(row=0)
input1.grid(row=0, column=1)
button1.grid(row=1, columnspan=2)

#making lables for the accuracy, prediction, and the std in the bottom frame
accurLabel = Label(botFrame, text="Accuracy:")
accurResLabelNb = Label(botFrame, textvariable = nbAccur)
predictLabel = Label(botFrame, text="Prediction:")
predictionResLabelNb = Label(botFrame, textvariable = nbPrediction)
stdLabelNb = Label(botFrame, text = "Standard Deviation:")
stdResLabelNb = Label(botFrame, textvariable = nbStd)
button2 = Button(botFrame, text="Show Data Distribution", fg="black", command=action_with_arg2)

#putting into the grids
accurLabel.grid(row=0)
accurResLabelNb.grid(row=0, column=1)
predictLabel.grid(row=0, column=2)
predictionResLabelNb.grid(row=0, column=3)
stdLabelNb.grid(row=0, column = 4)
stdResLabelNb.grid(row=0, column = 5)
button2.grid(row=1, columnspan = 10)

#size of your window app
root.geometry("700x150")
#making it not resizeable
root.resizable(0, 0)
#title of the windows(program)
root.title("Naive Bayes Prediction")

#mainloop is for making sure that the app running in the infinity loop so that the app runs till you close it
#with x button on the top right of the window
root.mainloop()

