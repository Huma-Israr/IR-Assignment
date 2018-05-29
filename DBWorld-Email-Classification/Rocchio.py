
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt

import numpy as np


#loading the matlab file
import scipy.io
mat = scipy.io.loadmat('C:\Users\Home\Desktop\IR Term Project\IR Project\dbworld emails dataset\MATLAB\dbworld_bodies')

mlables = mat['labels']  
mlabels=np.asarray(mlables) 
print("Shape of labels' set: ",mlabels.shape)

mdata = mat['inputs']  
mdata=np.asarray(mdata) 
print("Shape of data set: ",mdata.shape)


X_train, X_test, y_train, y_test = train_test_split(mdata, mlabels, test_size=0.25, random_state=55)

print ('X_train dimensions: ', X_train.shape)
print ('y_train dimensions: ', y_train.shape)
print ('X_test dimensions: ', X_test.shape)
print ('y_test dimensions: ', y_test.shape)


model = NearestCentroid().fit(X_train,y_train.ravel())


y_train_pred = model.predict(X_train) 
print("Training Data prediction: \n",y_train_pred)
print("Training Data ground truth: \n",y_train.ravel())
matrix = metrics.confusion_matrix(y_train, y_train_pred)
print(matrix)
accuracy = round((accuracy_score(y_train,y_train_pred))*100,2)
print("Accuracy for training dataset: ", accuracy,"%")
plt.matshow(matrix)
plt.title('Confusion Matrix for Train Data')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
y_test_pred = model.predict(X_test)
print("Testing Data Predicton: \n", y_test_pred)
print("Testing Data Ground Truth: \n", y_test.ravel())

matrix_test = confusion_matrix(y_test, y_test_pred)
print(matrix_test)
accuracy_test = (accuracy_score(y_test, y_test_pred))*100

print("Accuracy for Testing Dataset: ", accuracy_test,"%")

plt.matshow(matrix_test)
plt.title('Confusion Matrix for Test Data')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
