import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

#Loading the dataset
dataset = pd.read_csv('diabetes.csv')
print(len(dataset))
print(dataset)

 #Replace zeroes
zero_not_accepted = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'Insulin']
for column in zero_not_accepted:
    dataset[column] = dataset[column].replace(0, np.NaN)#pandas function to replace column of zeros with numpy NAN(NO DATA)
    mean = int(dataset[column].mean(skipna=True))#Calculate mean of dataset 
    dataset[column] = dataset[column].replace(np.NaN, mean)#replace mean with Mean value
print(dataset['Insulin'])


# split dataset
X = dataset.iloc[:, 0:8]
y = dataset.iloc[:, 8]
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=0 ,test_size=0.2)
print(len(X_train))
print(len(y_train))
print(len(X_test))
print(len(y_test))

#Feature scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#finding K value
import math
math.sqrt(len(y_test))

# Define the model: Init K-NN
classifier = KNeighborsClassifier(n_neighbors=11, p=2,metric='euclidean')

# Fit Model
classifier.fit(X_train, y_train)

# Predict the test set results
y_pred = classifier.predict(X_test)
y_pred

print(accuracy_score(y_test,y_pred))