from sklearn.datasets import fetch_lfw_people
import numpy as np
import matplotlib.pyplot as plt
from time import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


def show_orignal_images(pixels):
    # Displaying Orignal Images
    fig, axes = plt.subplots(6, 10, figsize=(11, 7),
                             subplot_kw={'xticks': [], 'yticks': []})
    for i, ax in enumerate(axes.flat):
        ax.imshow(np.array(pixels)[i].reshape(50,37), cmap='gray')
    plt.show()


def show_eigenfaces(pca):
    # Displaying Eigenfaces
    fig, axes = plt.subplots(3, 8, figsize=(9, 4),
                             subplot_kw={'xticks': [], 'yticks': []})
    for i, ax in enumerate(axes.flat):
        ax.imshow(pca.components_[i].reshape(50, 37), cmap='gray')
        ax.set_title("PC " + str(i + 1))
    plt.show()

lfw_people = fetch_lfw_people(min_faces_per_person= 120, resize=0.4)
x = lfw_people.data
y = lfw_people.target
show_orignal_images(x)

x_train, x_test, y_train, y_test = train_test_split(x, y)
pca = PCA(n_components=250).fit(x_train)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()
show_eigenfaces(pca)

x_train_pca = pca.transform(x_train)
clf = SVC(kernel='rbf',C=100,gamma='scale')
clf = clf.fit(x_train_pca, y_train)

print("Predicting people's names on the test set")
t0 = time()
x_test_pca = pca.transform(x_test)
y_pred = clf.predict(x_test_pca)
print("done in %0.3fs" % (time() - t0))
print(classification_report(y_test, y_pred))

clf = MLPClassifier(solver='lbfgs',alpha=0.00001,
                    hidden_layer_sizes=(1200, 200),random_state=1)
clf.fit(x_train_pca, y_train)
y_pred = clf.predict(x_test_pca)
print(classification_report(y_test, y_pred))

