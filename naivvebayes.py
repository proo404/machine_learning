import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from tensorflow.keras.datasets import mnist

(x_train,y_train), (x_test,y_test) = mnist.load_data()
x_train=x_train.reshape(60000,784)
x_test=x_test.reshape(10000,784)

plt.imshow(x_train[0].reshape((28,28)))

nb_model=GaussianNB()

fit_nb=nb_model.fit(x_train,y_train)
prediction=fit_nb.predict(x_test)

con_matrix= confusion_matrix(y_test,prediction)
print(con_matrix)

def diagonal_sum(con_matrix):
    sum=0
    for i in range(10):
        for j in range(10):
            if i==j:
                sum=sum+con_matrix[i,j]
    return sum
sum=diagonal_sum(con_matrix)
print(sum)
print('accuracy %:',sum/10000)
print(accuracy)