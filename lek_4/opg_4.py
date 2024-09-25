from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from matplotlib import pyplot as plt
from sklearn import decomposition
from sklearn import datasets
from sklearn import metrics
from sklearn import svm
import numpy as np

# getting the data.
faces = datasets.fetch_olivetti_faces()
faces.data.shape

# plot several images
fig = plt.figure(figsize=(8, 6))
for i in range(15):
    ax = fig.add_subplot(3, 5, i + 1, xticks=[], yticks=[])
    ax.imshow(faces.images[i], cmap=plt.cm.bone)
plt.savefig("face_001.png")

#making a test split
X_train, X_test, y_train, y_test = train_test_split(faces.data, faces.target, random_state=0)
print(X_train.shape, X_test.shape)

#pca
pca = decomposition.PCA(n_components=150, whiten=True)
pca.fit(X_train)

from matplotlib import pyplot as plt
fig = plt.figure(figsize=(8, 6))
plt.imshow(pca.mean_.reshape(faces.images[0].shape), cmap=plt.cm.bone)
plt.savefig("face_002.png")

print(pca.components_.shape)

from matplotlib import pyplot as plt
fig = plt.figure(figsize=(16, 6))
for i in range(30):
    ax = fig.add_subplot(3, 10, i + 1, xticks=[], yticks=[])
    ax.imshow(pca.components_[i].reshape(faces.images[0].shape), cmap=plt.cm.bone)
plt.savefig("face_003.png")

X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print(X_train_pca.shape)
print(X_test_pca.shape)

#Doing the Learning: Support Vector Machines
clf = svm.SVC(C=5., gamma=0.001)
clf.fit(X_train_pca, y_train)


from matplotlib import pyplot as plt
fig = plt.figure(figsize=(8, 6))
for i in range(15):
    ax = fig.add_subplot(3, 5, i + 1, xticks=[], yticks=[])
    ax.imshow(X_test[i].reshape(faces.images[0].shape), cmap=plt.cm.bone)
    y_pred = clf.predict(X_test_pca[i, np.newaxis])[0]
    color = ('black' if y_pred == y_test[i] else 'red')
    ax.set_title(y_pred, fontsize='small', color=color)
plt.savefig("face_004.png")


y_pred = clf.predict(X_test_pca)
print(metrics.classification_report(y_test, y_pred))
print(metrics.confusion_matrix(y_test, y_pred))



#Pipelining
print("\n\nPipelining")
from matplotlib import pyplot as plt
clf = Pipeline([('pca', decomposition.PCA(n_components=150, whiten=True)), ('svm', svm.LinearSVC(C=1.0))])
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(metrics.confusion_matrix(y_pred, y_test))
