import pandas as pd
from sklearn.pipeline import Pipeline
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
import nltk
nltk.download('stopwords')

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
