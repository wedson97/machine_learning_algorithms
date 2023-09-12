import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from yellowbrick.classifier import ConfusionMatrix

base = pd.read_csv('insurance.csv')
base = base.drop(columns=['Unnamed: 0'])

base.dropna(inplace=True)

Y = base.iloc[:,7]
X = base.iloc[:,[0,1,2,3,4,5,6,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]].values

labelEncoder = LabelEncoder()

for i in range(X.shape[1]):
    if X[:,i].dtype == 'object':
        X[:,i] = labelEncoder.fit_transform(X[:,i])

# X Variveis independentes
# Y Variaveis dependetes
X_treinamento, X_teste, Y_treinamento, Y_teste = train_test_split(X, Y, test_size=0.3, random_state=1)

print(Y_treinamento)
modelo = GaussianNB()
modelo.fit(X_treinamento,Y_treinamento)

previsoes = modelo.predict(X_teste)

accuracy = accuracy_score(Y_teste,previsoes)

precision = precision_score(Y_teste,previsoes,average='weighted')
recall = recall_score(Y_teste,previsoes, average='weighted')
f1 = f1_score(Y_teste,previsoes,average='weighted')

report = classification_report(Y_teste,previsoes)

confusao = ConfusionMatrix(modelo, classes=['Mild','Moderate','Severe'])
confusao.fit(X_treinamento,Y_treinamento)
confusao.score(X_teste,Y_teste)
confusao.poof()

print(f'Acuracia: {accuracy}, Precisão: {precision}, Recall: {recall}, f1: {f1}')
print(report)