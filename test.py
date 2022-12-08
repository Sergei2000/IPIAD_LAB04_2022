from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn import decomposition
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

n_clusters = 6
categories = ['comp.graphics', 'rec.autos','sci.crypt','soc.religion.christian','talk.politics.mideast']

twenty_train = fetch_20newsgroups(subset='train',categories = categories, shuffle=True, random_state=42)
NewsVectors = TfidfVectorizer()
#Generate features vectors
X_vectors = NewsVectors.fit_transform(twenty_train.data)

#reduce number of coordinates 
pca = PCA(n_components=2)
pca.fit(X_vectors.toarray())
X = pca.fit_transform(X_vectors.toarray())

print("Gm started")
gm = GaussianMixture(n_components=n_clusters, random_state=42)
gmprediction = gm.fit_predict(X)
print("Gm finished")


print("Kmeans start")
kmcluster =  KMeans(n_clusters=n_clusters,random_state=42)
kmprediction = kmcluster.fit_predict(X)
print("Kmeans finished")

# Get centers for each method
kmcenters =kmcluster.cluster_centers_
gmcenters =gm.means_

#draw plots
fig, axs = plt.subplots(2)
axs[0].scatter(x=X[:,0], y=X[:,1],c=kmprediction)
axs[0].scatter(x=kmcenters[:,0],y=kmcenters[:,1],marker='X',color="r")
axs[0].set_title("Clusterization with KMeans method:")
axs[1].scatter(x=X[:,0], y=X[:,1],c=gmprediction)
axs[1].scatter(x=gmcenters[:,0],y=gmcenters[:,1,],marker='X',color="r")
axs[1].set_title("Clusterization with GaussianMixture method:")
plt.show()










