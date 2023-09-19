from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
import matplotlib.pyplot as plt
from scipy.cluster._hierarchy import linkage

iris = datasets.load_iris()

Kmeans = KMeans(n_clusters=3,n_init='auto')
Kmeans.fit(iris.data)
print(Kmeans.labels_)

resultados = confusion_matrix(iris.target, Kmeans.labels_)
print(resultados)

dbscan = DBSCAN(eps=0.5, min_samples=3)
dbscan_labels = dbscan.fit_predict(iris.data)
print(dbscan_labels)

agglo = AgglomerativeClustering(n_clusters=3)
agglo_labels = agglo.fit_predict(iris.data)
print(agglo_labels)

def plot_clusters(data,labels,title):
    colors = ['red','green','purple','black']
    plt.figure(figsize=(8,4))
    for i,c,l in zip(range(-1,3), colors, ['noise', 'setosa', 'Versicolor','virginica']):
        if i == -1:
            plt.scatter(data[labels == i,0], data[labels == i,3], c=colors[i], label = l,alpha=0.5,s=50,marker='x')
        else:
            plt.scatter(data[labels == i,0], data[labels == i,3], c=colors[i], label = l,alpha=0.5,s=50)
    plt.legend()
    plt.title(title)
    plt.xlabel('Comprimento Sépala')
    plt.ylabel('Largura da Pétala')
    plt.show()

plot_clusters(iris.data, agglo_labels, 'Cluster hierarquico')
