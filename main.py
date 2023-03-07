from numba import njit
from time import time
import os 
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
import numpy as np
from time import time
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.datasets import load_digits
from SoftmaxRegression import SoftMaxRegression

def main():
    X,Y = load_digits(return_X_y=True)
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.25)

    X_train = np.array(X_train).T
    X_test = np.array(X_test).T


    ohe = OneHotEncoder()
    Y_train = ohe.fit_transform(Y_train.reshape(-1,1)).toarray().T
    Y_test = ohe.fit_transform(Y_test.reshape(-1,1)).toarray().T

    for i in ["Adam","momentum","BatchGrad"]:
        regression = SoftMaxRegression(X_train,Y_train,optimizer=i)
        regression.train(100000)

        print(f"The cost is :{regression.compute_cost()}")
        print(f"The accuracy is: {regression.predict(X_test,y_test)}")
        print()

if __name__ == "__main__":
    main()
