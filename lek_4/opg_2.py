import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression

iris = load_iris()

print(iris.data.shape)
print(iris.data[0])
print(iris.target_names)
print(iris.target)

#3.5.2.1
print("\n\n3.5.2.1 LinearRegression")
model = LinearRegression(n_jobs=1)
print(model)


x = np.array([0, 1, 2,  6])
y = np.array([0, 2, 4.1,11.9])


print(x)
X = x[:, np.newaxis]
print(X)

fit = model.fit(X, y)
print(model.coef_)


#3.6.2.2 knn
print("\n\n3.5.2.2 knn")
from sklearn import neighbors, datasets

knn = neighbors.KNeighborsClassifier(n_neighbors=1)
knn.fit(iris.data, iris.target)

print(iris.target_names[knn.predict([[3, 5, 4, 2]])])


#3.6.2.2 knn
print("\n\n3.5.2.2 knn")

from sklearn.datasets import load_digits
from matplotlib import pyplot as plt

digits = load_digits()

fig = plt.figure(figsize=(6, 6))  # figure size in inches
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

for i in range(64):
    ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
    ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')
    # label the image with the target value
    ax.text(0, 7, str(digits.target[i]))

plt.savefig("digits_plot.png")


#Plot a projection on the 2 first principal axis
print("\n\nPlot a projection on the 2 first principal axis")
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

fig = plt.figure(figsize=(6, 6))  # figure size in inches

pca = PCA(n_components=2)
proj = pca.fit_transform(digits.data)
plt.scatter(proj[:, 0], proj[:, 1], c=digits.target, cmap="Paired")
plt.colorbar()
plt.savefig("digits_plot_2.png")


### Classify with Gaussian naive Bayes

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics


X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target)

clf = GaussianNB()
clf.fit(X_train, y_train)

predicted = clf.predict(X_test)
expected = y_test

fig = plt.figure(figsize=(6, 6))  # figure size in inches
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

# visualizing the data.
for i in range(64):
    ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
    ax.imshow(X_test.reshape(-1, 8, 8)[i], cmap=plt.cm.binary, interpolation='nearest')
    if predicted[i] == expected[i]:
        ax.text(0, 7, str(predicted[i]), color='green')
    else:
        ax.text(0, 7, str(predicted[i]), color='red')

plt.savefig("digits_plot_3.png")

print("number, total number.")
print(X_test.shape)

matches = (predicted == expected)
print("number of correct matches")
print(matches.sum())

print(f"% correct:{matches.sum() / float(len(matches))}")
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))




