import csv
import pandas as pd
from sklearn.pipeline import Pipeline
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.svm.libsvm import predict_proba
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict


class Review:
    each_word_freq = {}

    def __init__(self):
        self.each_word_freq = {}


class TestReview:
    test_word_freq = {}

    def __init__(self):
        self.test_word_freq = {}


reviews = []
test_reviews = []

x_train = []
y_train = []
x_test = []
wordFreq = {}

data_set = pd.read_csv('data/imdb_train.txt', delimiter="\n", header=None).astype(str).values
test_set = pd.read_csv('data/imdb_test.txt', delimiter="\n", header=None).astype(str).values

for i in data_set:
    x_train.append(i[0][1:])
    y_train.append(int(i[0][0]))

for i in test_set:
    x_test.append(i[0][:])

# print(x_train[0].split(' '))
for row in x_train:
    new_review = Review()
    row = row.split(' ')
    for word in row:
        if word not in wordFreq:
            wordFreq[word] = 0
        wordFreq[word] += 1
        if word not in new_review.each_word_freq:
            new_review.each_word_freq[word] = 0
        new_review.each_word_freq[word] += 1
    reviews.append(new_review)

for row in x_test:
    new_review = TestReview()
    row = row.split(' ')
    for word in row:
        if word not in wordFreq:
            continue
        if word not in new_review.test_word_freq:
            new_review.test_word_freq[word] = 0
        new_review.test_word_freq[word] += 1
    test_reviews.append(new_review)

# The idea is to remove common word, such as the, a, is, was, and so on
# and also removing word that is not really affecting
delete_word = []
for word in wordFreq:
    if wordFreq[word] <= 900 or wordFreq[word] >= 5000:
        delete_word.append(word)

for i in delete_word:
    del wordFreq[i]

# print(len(wordFreq))

#######################################################################################
###             THIS CODE IS IMPORTANT TO CONVERT THE WORD INTO WEIGHT DATA         ###
###   UNCOMMENT IF ONLY WANT TO CREATE NEW TXT FILE, THOUGH THE FILE ALREADY GIVEN  ###
#######################################################################################

print("Train Data contstruction is on the go, it will take quite a long time")
# x=1
#
# for review in reviews:
#     new_array = []
#     for word in wordFreq:
#         if word not in review.each_word_freq:
#             new_array.append(0)
#         else:
#             new_array.append(int(review.each_word_freq[word]))
#     with open('x_train.csv', mode='a', newline='') as x_train_file:
#         train_writer = csv.writer(x_train_file, delimiter=',')
#         train_writer.writerow(new_array)
#     print(x)
#     x+=1
#
# x = 1
#
# for review in test_reviews:
#     new_array = []
#     for word in wordFreq:
#         if word not in review.test_word_freq:
#             new_array.append(0)
#         else:
#             new_array.append(int(review.test_word_freq[word]))
#     with open('x_test.csv', mode='a', newline='') as x_test_file:
#         test_writer = csv.writer(x_test_file, delimiter=',')
#         test_writer.writerow(new_array)
#     print(x)
#     x += 1

new_train_data = pd.read_csv("data/x_train.csv", header=None).astype(float).values
new_test_data = pd.read_csv("data/x_test.csv", header=None).astype(float).values

print("tensorflow performance might be slow, please wait a moment")


def create_model():
    model = Sequential()
    model.add(
        Dense(551, input_dim=551, kernel_initializer='normal',
              activation='relu'))  # relu to take out any negative value
    model.add(Dense(30, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_model, epochs=2, batch_size=5)))
pipeline = Pipeline(estimators)

pipeline = pipeline.fit(new_train_data, y_train)
predictions = pipeline.predict(new_test_data)

kfold = StratifiedKFold(n_splits=2, shuffle=True)
result = cross_val_predict(pipeline, new_test_data, predictions, cv=kfold)
# result = cross_val_predict(pipeline, new_train_data, y_train, cv=kfold)

#
# pipeline = pipeline.fit(new_train_data, y_train)
# predictions = pipeline.predict(new_test_data)
# print("Results: %.2f% % (%.2f% %)" % (result.mean() * 100, result.std() * 100))

number = 1

for prediction in result:
    print("Prediction of review ", number, " =", prediction)
    number += 1
