import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
N = 500 # Number of Samples
D = 2 #Dimensions

# Random data points
X = np.random.randn(N, D)

# Dividing above samples into 2 different gaussian blobs
X[ :250, : ] = X[ :250, : ] - 2*np.ones((250, D))
X[250:, : ] = X[250:, : ] + 2*np.ones((250, D))

# Creating targets for above 2 blobs
Y = np.array([0]*250 + [1]*250)

bias_term = np.ones((N, 1))

# Adding bias term to our data
Xbias = np.concatenate((X, bias_term), axis=1)

# randomly initialize the weights
Weights = np.random.randn(D+1)

# Dot Product
Z = Xbias.dot(Weights)

def sigmoid(Z):
    return 1/(1 + np.exp(-Z))

Yhat = sigmoid(Z)

# Our Cost Function for binary classification
# calculate the cross-entropy error
def cross_entropy(Y, Yhat):
    E = 0
    for i in range(len(Y)):
        if Y[i] == 1:
            E -= np.log(Yhat[i])
        else:
            E -= np.log(1 - Yhat[i])
    return E


# let's do gradient descent 100 times
learning_rate = 0.01
for i in range(200):
    if i % 10 == 0:
        print(cross_entropy(Y, Yhat))

    # gradient descent weight udpate
    Weights += learning_rate * Xbias.T.dot(Y - Yhat)

    # recalculate Yhat
    Yhat = sigmoid(Xbias.dot(Weights))

print(classification_report(Y, np.round(Yhat)))
print()
print(confusion_matrix(Y, np.round(Yhat)))
