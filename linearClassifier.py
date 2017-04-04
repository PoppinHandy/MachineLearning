# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 23:05:03 2017

@author: andyp

Using a pseudoinverse linear classifier that reduces mean-squared distance, classifies a machine failure dataset into 6 separate classes.
"""
import pandas as pd
import numpy as np

def getX(df):
    df.insert(0, "X0", 1)
    del df["Failure"]
    del df["Type"]
    X = np.array(df)
    return X

def combineColumns(df, colList, number):
    """Merges a specified number of columns into one column and deletes the separate columns."""
    for col in colList:
        index = df.columns.get_loc(col)
        df[col] = df[df.columns[index:index+number + 1]].apply(lambda x: ''.join(x.astype(str)),axis=1)
        df[col] = df[col].astype(int)
        for dc in df.columns.values[index + 1:index + number + 1]:
            del df[dc]
    return df

def deleteCols(df, colList):
    """Deletes all columns in a list from the dataframe."""
    for c in colList:
        del df[c]
    return df
    
def getWeights(X, T):
    """Returns an array of weights for determining linear classifier."""
    Xa_inverse = np.linalg.pinv(X)
    W = np.dot(Xa_inverse, T)
    return W
  
    
def signs(result):
    if result < 0:
        return -1
    else:
        return 1
    
def classify(testVector, weights, C=1):
    """Runs linear classifier based on number of classes and returns array of predicted class labels."""
    if (C == 1):
        result = np.dot(testVector, weights)
        classLabel = np.array([np.apply_along_axis(signs, 1, result)])
        classLabel = classLabel.T
        return classLabel
    else:
        result = np.dot(testVector, weights)
        maxIndexArray = list()
        for r in range(len(result)):
            maxIndex = np.argmax(result[r],axis=0)
            maxIndexArray.append(maxIndex)
        finalResult = np.array([maxIndexArray]).T
        return finalResult
  
def calculateBinaryMetrics(metricDict, results, trueLabels):
    for r in range(len(results)):
        if results[r] == 1:
            if results[r] == trueLabels[r]:
                metricDict["TP"] += 1
            elif results[r] != trueLabels[r]:
                metricDict["FP"] += 1
        elif results[r] == -1:
            if results[r] == trueLabels[r]:
                metricDict["TN"] += 1
            elif results[r] != trueLabels[r]:
                metricDict["FN"] += 1
    return metricDict
    
def calculateMultiClassMetrics(classes, results, trueLabels):
    metricDF = pd.DataFrame(0, columns=range(classes), index=range(classes))
    for r in range(len(results)):
        metricDF.iat[int(trueLabels[r]), int(results[r])] += 1
    for cols in metricDF.columns.values:
        ppv = metricDF.iat[cols, cols]/np.sum(metricDF[cols])
        print(cols, ": ", ppv)
    return metricDF
    
def printVector(vector):
    f = "printedVectors.txt"
    file = open(f, 'w')
    v = np.array_str(vector)
    v = v.replace("[", "").replace("]", "").replace(" ", "")
    file.write(v)
    file.close()
    
if __name__ == "__main__":
    excelLocation = "Docs/Assignment_4_Data_and_Template.xlsx"
    df = pd.read_excel(excelLocation)
    
    # Target columns
    multiclassT = np.array([df["Type"]])
    multiclassT = multiclassT.T
    numClasses = np.max(multiclassT) + 1
    binaryT = np.array([df["Failure"]])
    binaryT = binaryT.T
    
    # Weights for binary classifier
    X = getX(df)
    W = getWeights(X, binaryT)
    np.set_printoptions(threshold=2000, linewidth=120)

    # Turning multiclass labels into N x (number of classes) T target vector
    mctList = list()
    for c in multiclassT:
        temp= np.array([[-1]] * numClasses)
        temp[c] = 1
        mctList.append(temp.T)
    mcT = np.concatenate(mctList)
    multiclassW = getWeights(X, mcT)
    
    testData = pd.read_excel(excelLocation, sheetname="To be classified", skiprows=3)
    testX = getX(testData) 
    
    # Testing
    #binaryResult = classify(testX, W, 1)
    #multiClassResult = classify(testX, multiclassW, 6)
    
    # Calculating metrics
    binaryMetricDict = {"TP": 0, "FP": 0, "FN": 0, "TN":0}
    binaryMetricResult = classify(X, W, 1)
    multiClassResult = classify(X, multiclassW, 6)
    binaryMetric = calculateBinaryMetrics(binaryMetricDict, binaryMetricResult, binaryT)
    multiClassMetric = calculateMultiClassMetrics(6, multiClassResult, multiclassT)
    #print(multiClassMetric)