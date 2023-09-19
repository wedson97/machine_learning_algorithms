from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
import numpy as np

mtcars = pd.read_csv('mt_cars.csv')

X =mtcars[['mpg','hp']].values
Y =mtcars[['mpg']].values
print(X,Y)

kn = KNeighborsClassifier(n_neighbors=3)
model = kn.fit(X, Y)

y_pred = model.predict(X)
print(y_pred)

accuracy = accuracy_score(Y, y_pred)
precision = precision_score(Y,y_pred, average='weighted')
recall = recall_score(Y,y_pred, average='weighted')
f1 = f1_score(Y,y_pred, average='weighted')
cm = confusion_matrix(Y,y_pred)

print(f'acuracia: {accuracy}, Precisao: {precision}, recall: {recall}, f1: {f1}')
print('matriz de confusao:', cm)