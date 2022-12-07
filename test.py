from sklearn.feature_extraction.text import CountVectorizer
import nltk
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
from sklearn import decomposition

categories = ['comp.graphics', 'rec.autos','sci.crypt','soc.religion.christian','talk.politics.mideast']

twenty_train = fetch_20newsgroups(subset='train',categories = categories, shuffle=True, random_state=42)
count_vect = CountVectorizer()
X_vectors = count_vect.fit_transform(twenty_train.data)
pca = decomposition.PCA()
pca.fit(X_vectors.toarray())
X = pca.transform(X_vectors.toarray())
print(len(X[0]))
print("Gm started")
gm = GaussianMixture(n_components=4, random_state=0).fit(X[:1000])
gmprediction = gm.predict(X[1000:])
print("Gm finished")

from sklearn.cluster import KMeans

print("Kmeans start")
kmcluster = KMeans(n_clusters=4)
kmcluster.fit(X[:1000])
kmprediction = kmcluster.predict(X[1000:])
print("Kmeans finished")
print(kmprediction)
print(gmprediction)





