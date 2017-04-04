# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 16:25:02 2017

@author: andyp

This program is for the classification and anaylsis of the MNIST 10 Digit dataset. Runs dimension reduction using PCA, followed by k-means clustering and Bayesian classification on the dataset.
"""

import os, struct
import matplotlib as plt
import scipy.sparse as sparse
import scipy.linalg as linalg

from array import array as pyarray
from numpy import append, array, int8, uint8, zeros as np
from pylab import *
from numpy import *
import numpy.linalg as LA
import pandas as pd

def load_mnist(dataset="training", digits=range(10), path='D:\\MachineLearning\\UCSC Course\\HW3_TrainingSet'):
    
    """
    Adapted from: http://cvxopt.org/applications/svm/index.html?highlight=mnist
    """

    if dataset == "training":
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset == "testing":
        fname_img = os.path.join(path, 't10k-images.idx')
        fname_lbl = os.path.join(path, 't10k-labels.idx')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    ind = [ k for k in range(size) if lbl[k] in digits ]
    N = len(ind)

    images = zeros((N, rows, cols), dtype=uint8)
    labels = zeros((N, 1), dtype=int8)
    for i in range(len(ind)):
        images[i] = array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]).reshape((rows, cols))
        labels[i] = lbl[ind[i]]

    return images, labels

def displayImg(X, amount):
    if (X.shape[1] == 784):
        for s in range(amount):
            plt.imshow(X[s].reshape(28, 28), interpolation='None', cmap=cm.gray)
            plt.show()
    else:
        print("Dimension is not 784.")
        
def getX(digits):    
    images, labels = load_mnist('training', digits=digits)
    # converting from NX28X28 array into NX784 array
    flatimages = list()
    for i in images:
        flatimages.append(i.ravel())
    X = np.asarray(flatimages)
    return X, [images, labels]
 
def getZ(X):
    """Returns mean subtracted from X matrix."""
    mean_vector = np.mean(X, axis=0)
    Z = X - mean_vector
    checkZ(Z, X, mean_vector, 0, 255)
    return Z, mean_vector

def checkZ(Z, X, mean_vector, minimum, maximum):
    m1 = np.amin(mean_vector)
    m2 = np.amax(mean_vector)
    zMean = np.mean(Z, axis=0)
    mz1 = np.amin(zMean)
    mz2 = np.amax(zMean)
    
    for x in zMean:
        if round(x) != 0:
            print("Mean of Z is not a vector of 0s. ", x)
            return False
            
    if (mean_vector.shape !=  (X.shape[1], )):
        print("Your mean vector is not the correct shape. ", mean_vector.shape)
        return False
    elif (m1 < minimum) or (m2 > maximum):
        print("Mean vector max and mins are outside the bounds. ", "Min: ", m1, " Max: ", m2)
        return False
    elif (Z.shape != X.shape):
        print("Z dimensions are wrong. ", Z.shape)
        return False
    elif (mz1 > 0) or (mz2 < 0):
        print("Mean of Z min and max are out of bounds. ", "Min: ", mz1, " Max: ", mz2)
        return False
    else:
        return True
    
def getC(Z):
    """Returns covariance matrix given mean subtracted from feature matrix"""
    C = np.cov(Z, rowvar=False)
    #print("Shape of C is:", C.shape)
    
#==============================================================================
#     # Checking for symmetry
#     for row in range(len(C.shape)):
#         if C[row][row] < 0:
#             print(C[row][row], "is negative!")
#         for col in range(len(C[0].shape)):
#             if C[row][col] != C[col][row]:
#                 print(C[row][col], "is not equal to", C[col][row])
#==============================================================================
    return C
    
def getV(C):
    """Returns eigenvector matrices based on covariance matrix. The rows represent eigenvectors."""
    [theta, V] = LA.eigh(C)
    V = np.flipud(V.T)  # Transposed because python formats columns as eigenvectors
    theta = np.flipud(theta)    # Theta are eigenvalues
    
    # CHECKING FOR NORMALIZATION
#==============================================================================
#     for r in range(10):
#         s = np.power(V[r], 2)
#         totalS = np.sum(s)
#         q = np.sqrt(totalS)
#         print (q)
#==============================================================================
#==============================================================================
#     pairs = [[11,12], [13, 17], [14, 10]]
#     for p in pairs:
#         multiplied = np.multiply(V[p[0]], V[p[1]])
#         print(np.sum(multiplied))
#==============================================================================
    return [theta, V]

def getP(V, Z):
    """Returns principal components matrix given eigenvectors. Columns are principal components"""
    P = np.dot(Z, V.T)
    #print(P.shape)
    return P
    
def getPrincipalComponents(P, V, number):
    """Returns the chosen number of principal components to do reduction on."""
    PC = np.dot(P[:, 0:number], V[0:number, :])
    #print(PC.shape)
    return PC
    
def printVector(vector):
    f = "printedVectors.txt"
    file = open(f, 'w')
    for v in vector:
        v = str(v) + ","
        #v = v.replace("[", "").replace("]", "")
        #file.write(np.array_str(v))
        file.write(v)
    file.write("\n\n")
    file.close()

def plotPC(PC1, PC2, labelList):
    """Plots a scatter plot of the any 2 specified dimensions after running PCA."""
    pc1 = [[],[],[],[],[],[],[],[],[],[]]
    pc2 = [[],[],[],[],[],[],[],[],[],[]]
    for l in range(len(labelList)):
        # l returns a number within a numpy array
        actualNum = labelList[l][0]
        pc1[actualNum].append(PC1[l])
        pc2[actualNum].append(PC2[l])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    colorList = ["red", "green", "blue", "black", "gray", "yellow", "cyan", "magenta", "burlywood", "purple"]
    for count in range(10):
        plt.scatter(pc1[count], pc2[count], c=colorList[count], lw=0, label = str(count))
    plt.legend(scatterpoints = 1 )
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    fig.savefig("2D_10MNistGraph.png")
    plt.close()
    
def createBayesianPDF(query, X):
    """Creates a probability density function and uses it to return the probability of the query being in a label class."""
    X = np.asarray(X)
    sampleSize = X.shape[0]
    muX = np.mean(X, axis=0)
    z = X - muX
    c = np.cov(z, rowvar=False)
    diff = np.array([query - muX])
    diffT = diff.transpose()
    detX = np.linalg.det(c)
    
    matrix = (-0.5) * (np.dot(np.dot(diff, np.linalg.inv(c)), diffT))
    fraction = sampleSize/(np.sqrt(detX))
    prob = fraction * np.power(np.e, matrix)
    return prob[0][0]

def classifyBayesian(query, X, label):
    """Using the Bayesian PDF for a normal distribution, returns the class that the query belongs to based on probability."""
    numMatch = [[],[],[],[],[],[],[],[],[],[]]
    probabilities = list()
    for d in range(10):
        m = np.where(label == d)
        for match in m[0]:
            numMatch[d].append(X[match])
        probabilities.append(createBayesianPDF(query, numMatch[d]))
    
    probSum = np.sum(probabilities)
    maximum = -1
    endInt = -1
    for d2 in range(10):
        prob = probabilities[d2]/probSum
        if prob > maximum:
            endInt = d2
            maximum = prob
    return [endInt, maximum]
    
def createHistogram(bins, positiveData, negativeData, minmax1, minmax2):
    # Visible Deprecation Warning 
    bins = int(bins)
    positiveHisto = np.zeros((bins, bins), dtype=int)
    negativeHisto = np.zeros((bins, bins), dtype=int)
    
    c1_Range = minmax1[1] - minmax1[0]
    c2_Range = minmax2[1] - minmax2[0]
    
    # Positive Data
    for p in positiveData:
        rowP = int(round((bins - 1) * ((p[0] - minmax1[0])/c1_Range)))
        colP = int(round((bins - 1) * ((p[1] - minmax2[0])/c2_Range)))
        positiveHisto[rowP][colP] += 1 
    
    # Negative Data
    for n in negativeData:
        rowN = int(round((bins - 1) * ((n[0] - minmax1[0])/c1_Range)))
        colN = int(round((bins - 1) * ((n[1] - minmax2[0])/c2_Range)))
        negativeHisto[rowN][colN] += 1 
    
#==============================================================================
#     for r in range(len(negativeHisto)):
#        for c in range(negativeHisto.shape[0]):
#            print(negativeHisto[r][c], end = " ")
#        print()
#==============================================================================
       
    return [positiveHisto, negativeHisto]

   
def classifyHistogram(bins, query, histogram, minmax1, minmax2):
    positiveHisto = histogram[0]
    negativeHisto = histogram[1]
    
    c1_Range = minmax1[1] - minmax1[0]
    c2_Range = minmax2[1] - minmax2[0]
    
    row = int(round((bins - 1) * ((query[0] - minmax1[0])/c1_Range)))
    col = int(round((bins - 1) * ((query[1] - minmax2[0])/c2_Range)))
    
    if row < 0:
        return 0
    elif col < 0:
        return 0
    elif row > 24:
        return 0
    elif col > 24:
        return 0
    
    if (positiveHisto[row][col] + negativeHisto[row][col]) != 0:    
        probP = positiveHisto[row][col]/(positiveHisto[row][col] + negativeHisto[row][col])
        probN = negativeHisto[row][col]/(positiveHisto[row][col] + negativeHisto[row][col])
        if probP > probN:
            return 1
        elif probP < probN:
            return -1
        else:
            return 0
    else:
        return 0
#==============================================================================
#     for r in range(len(bins)):
#         for c in range(len(bins)):
#             if (positiveHisto[r][c] + negativeHisto[r][c]) == 0:
#                 undecidable += 1
#             else:
#                 probP = positiveHisto[r][c]/(positiveHisto[r][c] + negativeHisto[r][c])
#                 probN = 1 - probP
#==============================================================================

def num_of_Components(eigenValues, threshold=80):
    """For a specified accuracy, returns the amount of principal components needed."""
    sumEigen = np.sum(eigenValues)
    eigenEffectiveness = 100*(np.cumsum(eigenValues)/sumEigen)
    fig = plt.figure()
    ax = fig.gca()
    plt.plot(eigenEffectiveness)
    ax.set_xlabel("Index")
    ax.set_ylabel("Percent of Variance Explained")
    fig.savefig("ThresholdGraph.png")
    plt.close()
    for e in range(len(eigenEffectiveness)):
        if eigenEffectiveness[e] >= threshold:
            return e
    
def runPCA(X, labelList, threshold=80):
    """Runs PCA"""
    Z, mean_vector = getZ(X)
    C = getC(Z)
    theta, V = getV(C)
    P = getP(V, Z)
    #amount = num_of_Components(theta, threshold)
    #twoPrincipals = P[:, 0:num]
    #plotPC(P[:,0:1], P[:,1:2], labelList)
    return P

def kMeans(X, K=10, iterationLimit=1000, tol=10e-6):
    """Runs kmeans clustering."""
    # X shape is (10, 43)
    muK = list()
    
    # Initializing d-dimensional random mean data
    for i in range(K):
        randomMean = list()
        for j in range(X.shape[1]):
            jMin = np.min(X[:, j])
            jMax = np.max(X[:, j])
            randomMean.append(np.random.uniform(jMin, jMax))
        muK.append(randomMean)
    muK = np.array(muK) # (10, 43)
    
    # Initializing class labels and other parameters
    c = np.zeros([X.shape[0], 1])
    c = c.astype(int)
    error = 10000
    iterations = 0
    
    while(iterations < iterationLimit and error > tol):
        # featureVector is number of observations
        # Assign points to a class label based on distance to centroid points
        for featureVector in range(X.shape[0]):
            kmin = math.inf
            kIndex = 0
            
            # kr = clusters number
            for kr in range(K):
                dist = np.power(X[featureVector] - muK[kr], 2)
                meanSum = dist.sum()
                if (meanSum < kmin):
                    kmin = meanSum
                    kIndex = kr
            c[featureVector][0] = kIndex
            
        # Update Means Step
        tmpError = 0
        for kr2 in range(K):
            countIndex = np.where(c == kr2)
            if (countIndex[0].shape[0] > 0):
                indexMatch = countIndex[0]
                finalList = X[indexMatch]
                #muK2 = np.sum(finalList, axis=0)/np.sum(finalList)
                muK2 = np.mean(finalList, axis=0)
                tmpError2 = np.max(np.absolute(muK[kr2] - muK2))
                if (tmpError < tmpError2):
                    tmpError = tmpError2
                muK[kr2] = muK2
        error = tmpError
        iterations += 1
        print(iterations, " ", error)
    return c
        
def calculateMultiClassMetrics(classes, results, trueLabels):
    """Calculates the PPV and accuracy values of each digit using a confusion matrix."""
    metricDF = pd.DataFrame(0, columns=range(classes), index=range(classes))
    for r in range(len(results)):
        metricDF.iat[int(trueLabels[r]), int(results[r])] += 1
    for cols in metricDF.columns.values:
        ppv = metricDF.iat[cols, cols]/np.sum(metricDF[cols])
        print("PPV: ", cols, ": ", ppv)
        tempDF = metricDF.copy()
        del tempDF[cols]
        tempDF = tempDF.drop([cols])
        TN = tempDF.values.sum()
        TP = df.iat[cols, cols]
        everything = metricDF.values.sum()
        accuracy = (TP + TN)/everything
        print("Accuracy: ", cols, ": ", accuracy)
    return metricDF
     
if __name__ == "__main__":

    X, imageAndlabelList = getX(range(10))
    labelList = np.array(imageAndlabelList[1])
    P = runPCA(X, labelList)
    pc = P[:, 0:44] # 44 is the optimum number of components to achieve ~80% accuracy
    
    # Uncomment to use Bayesian classifier
#==============================================================================
#     totalSample = pc.shape[0]
#     accuracyBayesian = 0
#     for pcQuery in range(len(pc)):
#         bayesianResults = classifyBayesian(pc[pcQuery], pc, labelList)
#         if (bayesianResults[0] == labelList[pcQuery]):
#             accuracyBayesian += 1
#     print("Accuracy of Bayesian: ", accuracyBayesian/totalSample)
#==============================================================================

    # Uncomment to use k-means classifier
#==============================================================================
#     results = kMeans(pc, 10)
#     df = calculateMultiClassMetrics(10, results, labelList)
#     df.to_csv("myresults.csv")
#==============================================================================
    