from sklearn import metrics
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


digits = load_digits()

for i in range(1,9,1):
    print(f"\nThe test split in this configuration is {i/10}")
    X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target,  test_size=(i/10))
    X = X_train 
    y = y_train #digits.target

    # Instantiate and train the classifier
    from sklearn.neighbors import KNeighborsClassifier
    clf = KNeighborsClassifier(n_neighbors=1)
    clf.fit(X, y) 
    # Check the results using metrics
    y_pred = clf.predict(X_test)

    print(metrics.confusion_matrix(y_pred, y_test))
